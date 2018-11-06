// Copyright 2008 Dolphin Emulator Project
// Licensed under GPLv2+
// Refer to the license.txt file included.

#include "VideoCommon/Fifo.h"

#include <atomic>
#include <cstring>

#include "Common/Assert.h"
#include "Common/Atomic.h"
#include "Common/BlockingLoop.h"
#include "Common/ChunkFile.h"
#include "Common/Event.h"
#include "Common/FPURoundMode.h"
#include "Common/MemoryUtil.h"
#include "Common/MsgHandler.h"

#include "Core/ConfigManager.h"
#include "Core/CoreTiming.h"
#include "Core/HW/Memmap.h"
#include "Core/Host.h"

#include "VideoCommon/AsyncRequests.h"
#include "VideoCommon/CPMemory.h"
#include "VideoCommon/CommandProcessor.h"
#include "VideoCommon/DataReader.h"
#include "VideoCommon/OpcodeDecoding.h"
#include "VideoCommon/Statistics.h"
#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoBackendBase.h"
#include "VideoCommon/XFMemory.h"
namespace Fifo
{
static constexpr u32 FIFO_SIZE = 2 * 1024 * 1024;
static constexpr int GPU_TIME_SLOT_SIZE = 1000;

static Common::BlockingLoop s_gpu_mainloop;

static Common::Flag s_emu_running_state;

// Most of this array is unlikely to be faulted in...
static u8 s_fifo_aux_data[FIFO_SIZE];
static u8* s_fifo_aux_write_ptr;
static u8* s_fifo_aux_read_ptr;

// This could be in SConfig, but it depends on multiple settings
// and can change at runtime.
static bool s_use_deterministic_gpu_thread;

static CoreTiming::EventType* s_event_sync_gpu;

// STATE_TO_SAVE
static u8* s_video_buffer;
static u8* s_video_buffer_read_ptr;
static std::atomic<u8*> s_video_buffer_write_ptr;
static std::atomic<u8*> s_video_buffer_seen_ptr;
static u8* s_video_buffer_pp_read_ptr;
// The read_ptr is always owned by the GPU thread.  In normal mode, so is the
// write_ptr, despite it being atomic.  In deterministic GPU thread mode,
// things get a bit more complicated:
// - The seen_ptr is written by the GPU thread, and points to what it's already
// processed as much of as possible - in the case of a partial command which
// caused it to stop, not the same as the read ptr.  It's written by the GPU,
// under the lock, and updating the cond.
// - The write_ptr is written by the CPU thread after it copies data from the
// FIFO.  Maybe someday it will be under the lock.  For now, because RunGpuLoop
// polls, it's just atomic.
// - The pp_read_ptr is the CPU preprocessing version of the read_ptr.

static std::atomic<int> s_sync_ticks;
static bool s_syncing_suspended;
static Common::Event s_sync_wakeup_event;

void DoState(PointerWrap& p)
{
  p.DoArray(s_video_buffer, FIFO_SIZE);
  u8* write_ptr = s_video_buffer_write_ptr;
  p.DoPointer(write_ptr, s_video_buffer);
  s_video_buffer_write_ptr = write_ptr;
  p.DoPointer(s_video_buffer_read_ptr, s_video_buffer);
  if (p.mode == PointerWrap::MODE_READ && s_use_deterministic_gpu_thread)
  {
    // We're good and paused, right?
    s_video_buffer_seen_ptr = s_video_buffer_pp_read_ptr = s_video_buffer_read_ptr;
  }

  p.Do(s_sync_ticks);
  p.Do(s_syncing_suspended);
}

void PauseAndLock(bool doLock, bool unpauseOnUnlock)
{
  if (doLock)
  {
    SyncGPU(SyncGPUReason::Other);
    EmulatorState(false);

    const SConfig& param = SConfig::GetInstance();

    if (!param.bCPUThread || s_use_deterministic_gpu_thread)
      return;

    s_gpu_mainloop.WaitYield(std::chrono::milliseconds(100), Host_YieldToUI);
  }
  else
  {
    if (unpauseOnUnlock)
      EmulatorState(true);
  }
}

void Init()
{
  // Padded so that SIMD overreads in the vertex loader are safe
  s_video_buffer = static_cast<u8*>(Common::AllocateMemoryPages(FIFO_SIZE + 4));
  ResetVideoBuffer();
  if (SConfig::GetInstance().bCPUThread)
    s_gpu_mainloop.Prepare();
  s_sync_ticks.store(0);
}

void Shutdown()
{
  if (s_gpu_mainloop.IsRunning())
    PanicAlert("Fifo shutting down while active");

  Common::FreeMemoryPages(s_video_buffer, FIFO_SIZE + 4);
  s_video_buffer = nullptr;
  s_video_buffer_write_ptr = nullptr;
  s_video_buffer_pp_read_ptr = nullptr;
  s_video_buffer_read_ptr = nullptr;
  s_video_buffer_seen_ptr = nullptr;
  s_fifo_aux_write_ptr = nullptr;
  s_fifo_aux_read_ptr = nullptr;
}

// May be executed from any thread, even the graphics thread.
// Created to allow for self shutdown.
void ExitGpuLoop()
{
  // This should break the wait loop in CPU thread
  CommandProcessor::fifo.bFF_GPReadEnable = false;
  FlushGpu();

  // Terminate GPU thread loop
  s_emu_running_state.Set();
  s_gpu_mainloop.Stop(s_gpu_mainloop.kNonBlock);
}

void EmulatorState(bool running)
{
  s_emu_running_state.Set(running);
  if (running)
    s_gpu_mainloop.Wakeup();
  else
    s_gpu_mainloop.AllowSleep();
}

void SyncGPU(SyncGPUReason reason, bool may_move_read_ptr)
{
  if (s_use_deterministic_gpu_thread)
  {
    s_gpu_mainloop.Wait();
    if (!s_gpu_mainloop.IsRunning())
      return;

    // Opportunistically reset FIFOs so we don't wrap around.
    if (may_move_read_ptr && s_fifo_aux_write_ptr != s_fifo_aux_read_ptr)
      PanicAlert("aux fifo not synced (%p, %p)", s_fifo_aux_write_ptr, s_fifo_aux_read_ptr);

    memmove(s_fifo_aux_data, s_fifo_aux_read_ptr, s_fifo_aux_write_ptr - s_fifo_aux_read_ptr);
    s_fifo_aux_write_ptr -= (s_fifo_aux_read_ptr - s_fifo_aux_data);
    s_fifo_aux_read_ptr = s_fifo_aux_data;

    if (may_move_read_ptr)
    {
      u8* write_ptr = s_video_buffer_write_ptr;

      // what's left over in the buffer
      size_t size = write_ptr - s_video_buffer_pp_read_ptr;

      memmove(s_video_buffer, s_video_buffer_pp_read_ptr, size);
      // This change always decreases the pointers.  We write seen_ptr
      // after write_ptr here, and read it before in RunGpuLoop, so
      // 'write_ptr > seen_ptr' there cannot become spuriously true.
      s_video_buffer_write_ptr = write_ptr = s_video_buffer + size;
      s_video_buffer_pp_read_ptr = s_video_buffer;
      s_video_buffer_read_ptr = s_video_buffer;
      s_video_buffer_seen_ptr = write_ptr;
    }
  }
}

void PushFifoAuxBuffer(const void* ptr, size_t size)
{
  if (size > (size_t)(s_fifo_aux_data + FIFO_SIZE - s_fifo_aux_write_ptr))
  {
    SyncGPU(SyncGPUReason::AuxSpace, /* may_move_read_ptr */ false);
    if (!s_gpu_mainloop.IsRunning())
    {
      // GPU is shutting down
      return;
    }
    if (size > (size_t)(s_fifo_aux_data + FIFO_SIZE - s_fifo_aux_write_ptr))
    {
      // That will sync us up to the last 32 bytes, so this short region
      // of FIFO would have to point to a 2MB display list or something.
      PanicAlert("absurdly large aux buffer");
      return;
    }
  }
  memcpy(s_fifo_aux_write_ptr, ptr, size);
  s_fifo_aux_write_ptr += size;
}

void* PopFifoAuxBuffer(size_t size)
{
  void* ret = s_fifo_aux_read_ptr;
  s_fifo_aux_read_ptr += size;
  return ret;
}

// Description: RunGpuLoop() sends data through this function.
/*static u32 ReadDataFromFifo(u32 readPtr, u32 len = 32)
{
  if (len > (s_video_buffer + FIFO_SIZE - s_video_buffer_write_ptr))
  {
    size_t existing_len = s_video_buffer_write_ptr - s_video_buffer_read_ptr;
    if (len > (FIFO_SIZE - existing_len))
    {
      PanicAlert("FIFO out of bounds (existing %zu + new %zu > %u)", existing_len, len, FIFO_SIZE);
      return 0;
    }
    memmove(s_video_buffer, s_video_buffer_read_ptr, existing_len);
    s_video_buffer_write_ptr = s_video_buffer + existing_len;
    s_video_buffer_read_ptr = s_video_buffer;
  }
  // Copy new video instructions to s_video_buffer for future use in rendering the new picture
  Memory::CopyFromEmu(s_video_buffer_write_ptr, readPtr, len);
  s_video_buffer_write_ptr += len;
  return len;
}*/

// The deterministic_gpu_thread version.
static u32 ReadDataFromFifoOnCPU(u32 readPtr, u32 len = 32)
{
  u8* write_ptr = s_video_buffer_write_ptr;
  if (len > (s_video_buffer + FIFO_SIZE - write_ptr))
  {
    // We can't wrap around while the GPU is working on the data.
    // This should be very rare due to the reset in SyncGPU.
    SyncGPU(SyncGPUReason::Wraparound);
    if (!s_gpu_mainloop.IsRunning())
    {
      // GPU is shutting down, so the next asserts may fail
      return 0;
    }

    if (s_video_buffer_pp_read_ptr != s_video_buffer_read_ptr)
    {
      PanicAlert("desynced read pointers");
      return 0;
    }
    write_ptr = s_video_buffer_write_ptr;
    size_t existing_len = write_ptr - s_video_buffer_pp_read_ptr;
    if (len > (size_t)(FIFO_SIZE - existing_len))
    {
      PanicAlert("FIFO out of bounds (existing %zu + new %zu > %u)", existing_len, len, FIFO_SIZE);
      return 0;
    }
  }
  Memory::CopyFromEmu(s_video_buffer_write_ptr, readPtr, len);
  s_video_buffer_pp_read_ptr = OpcodeDecoder::Run<true>(
      DataReader(s_video_buffer_pp_read_ptr, write_ptr + len), nullptr, false);
  // This would have to be locked if the GPU thread didn't spin.
  s_video_buffer_write_ptr = write_ptr + len;
  return len;
}

void ResetVideoBuffer()
{
  s_video_buffer_read_ptr = s_video_buffer;
  s_video_buffer_write_ptr = s_video_buffer;
  s_video_buffer_seen_ptr = s_video_buffer;
  s_video_buffer_pp_read_ptr = s_video_buffer;
  s_fifo_aux_write_ptr = s_fifo_aux_data;
  s_fifo_aux_read_ptr = s_fifo_aux_data;
}

static void FifoFlushMemory(u8* start_ptr)
{
  if (start_ptr >= s_video_buffer && start_ptr <= s_video_buffer + FIFO_SIZE)
  {
    s_video_buffer_read_ptr = start_ptr;
  }
  else
  {
    CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;
    u8* init_ptr = Memory::GetPointer(fifo.CPReadPointer);
    int used_size = (int)(start_ptr - init_ptr);
    fifo.CPReadPointer += used_size;
    fifo.CPReadWriteDistance -= used_size;
  }
}

bool FifoReserveMemory(u8*& start_ptr, u8*& end_ptr, u32 need_size)
{
  if (end_ptr - start_ptr >= need_size)
  {
    return true;
  }

  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;

  ASSERT_MSG(COMMANDPROCESSOR, (s32)fifo.CPReadWriteDistance >= 0,
             "Negative fifo.CPReadWriteDistance = %i in FIFO Loop !\nThat can produce "
             "instability in the game. Please report it.",
             fifo.CPReadWriteDistance);

  u8* init_ptr;
  bool copy_all_data;
  bool copy_to_video_buffer = false;
  if (start_ptr == end_ptr)
  {
    if (start_ptr >= s_video_buffer && start_ptr <= s_video_buffer + FIFO_SIZE)
    {
      s_video_buffer_read_ptr = s_video_buffer_write_ptr;
    }
    // non used data, don't copy
    copy_all_data = false;
    init_ptr = start_ptr;
  }
  else if (start_ptr >= s_video_buffer && start_ptr <= s_video_buffer + FIFO_SIZE)
  {
    // used data already in video buffer, copy remaining data to video buffer
    copy_to_video_buffer = true;
    copy_all_data = false;
    init_ptr = s_video_buffer_read_ptr;
  }
  else
  {
    // there is used data in fifo memory, copy all data to video buffer
    copy_all_data = true;
    init_ptr = Memory::GetPointer(fifo.CPReadPointer);
  }

  bool result;
  u32 used_size = start_ptr - init_ptr;
  u32 total_size = used_size + need_size;
  u32 page_size = fifo.CPEnd - fifo.CPReadPointer + 32;
  u8* write_ptr = s_video_buffer_write_ptr;

  if (total_size > fifo.CPReadWriteDistance)
  {
    total_size = fifo.CPReadWriteDistance;
    result = false;
    copy_to_video_buffer = true;
  }
  else
  {
    // align to 32 bytes
    total_size = (total_size + 0x1F) & ~0x1F;
    need_size = (need_size + 0x1F) & ~0x1F;
    if (total_size > fifo.CPReadWriteDistance)
    {
      total_size = fifo.CPReadWriteDistance;
    }
    if (need_size > fifo.CPReadWriteDistance)
    {
      need_size = fifo.CPReadWriteDistance;
    }
    result = true;
  }

  if (page_size < total_size)
  {
    u32 copy_size = copy_all_data ? total_size : need_size;
    // need size is bigger than current fifo memory page,
    // so copy from fifo memory to video buffer
    u32 remain_size = copy_size - page_size;
    u32 buff_size = s_video_buffer + FIFO_SIZE - write_ptr;
    if (buff_size < copy_size)
    {
      // move memory
      u32 data_size = write_ptr - s_video_buffer_read_ptr;
      memmove(s_video_buffer, s_video_buffer_read_ptr, data_size);
      s_video_buffer_read_ptr = s_video_buffer;
      write_ptr = s_video_buffer + data_size;
    }

    // read current page
    Memory::CopyFromEmu(write_ptr, fifo.CPReadPointer, page_size);
    fifo.CPReadPointer = fifo.CPBase;
    fifo.CPReadWriteDistance -= page_size;
    write_ptr += page_size;

    // read next page
    Memory::CopyFromEmu(write_ptr, fifo.CPBase, remain_size);
    fifo.CPReadPointer += remain_size;
    fifo.CPReadWriteDistance -= remain_size;
    write_ptr += remain_size;

    s_video_buffer_write_ptr = write_ptr;
    start_ptr = s_video_buffer_read_ptr + used_size;
    end_ptr = write_ptr;
  }
  else if (copy_to_video_buffer)
  {
    u32 copy_size = copy_all_data ? total_size : need_size;
    // process remain data in video buffer
    u32 buff_size = s_video_buffer + FIFO_SIZE - write_ptr;
    if (buff_size < copy_size)
    {
      // move memory
      u32 data_size = write_ptr - s_video_buffer_read_ptr;
      memmove(s_video_buffer, s_video_buffer_read_ptr, data_size);
      s_video_buffer_read_ptr = s_video_buffer;
      write_ptr = s_video_buffer + data_size;
    }

    Memory::CopyFromEmu(write_ptr, fifo.CPReadPointer, copy_size);
    fifo.CPReadPointer += copy_size;
    fifo.CPReadWriteDistance -= copy_size;
    write_ptr += copy_size;
    s_video_buffer_write_ptr = write_ptr;
    start_ptr = s_video_buffer_read_ptr + used_size;
    end_ptr = write_ptr;
  }
  else
  {
    // use fifo memory directly
    start_ptr = Memory::GetPointer(fifo.CPReadPointer + used_size);
    end_ptr = Memory::GetPointer(fifo.CPReadPointer + total_size);
  }

  return result;
}

void OpcodeDecoderRun(int& cycles)
{
  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;

  int refarray;
  int totalCycles = 1;
  u8* start_ptr = s_video_buffer_read_ptr;
  u8* end_ptr = s_video_buffer_write_ptr;
  while (true)
  {
    if (!FifoReserveMemory(start_ptr, end_ptr, 1))
      goto end;

    u8 cmd_byte = *start_ptr;
    start_ptr += 1;

    switch (cmd_byte)
    {
    case OpcodeDecoder::GX_NOP:
      totalCycles += 6;  // Hm, this means that we scan over nop streams pretty slowly...
      break;

    case OpcodeDecoder::GX_UNKNOWN_RESET:
      totalCycles += 6;  // Datel software uses this command
      DEBUG_LOG(VIDEO, "GX Reset?: %08x", cmd_byte);
      break;

    case OpcodeDecoder::GX_LOAD_CP_REG:
    {
      if (!FifoReserveMemory(start_ptr, end_ptr, 1 + 4))
        goto end;
      totalCycles += 12;
      u8 sub_cmd = *start_ptr;
      start_ptr += 1;
      u32 value = Common::FromBigEndian<u32>(*((u32*)start_ptr));
      start_ptr += 4;
      LoadCPReg(sub_cmd, value, false);
      INCSTAT(stats.thisFrame.numCPLoads);
    }
    break;

    case OpcodeDecoder::GX_LOAD_XF_REG:
    {
      if (!FifoReserveMemory(start_ptr, end_ptr, 4))
        goto end;
      u32 Cmd2 = Common::FromBigEndian<u32>(*((u32*)start_ptr));
      start_ptr += 4;
      int transfer_size = ((Cmd2 >> 16) & 15) + 1;
      if (!FifoReserveMemory(start_ptr, end_ptr, transfer_size * sizeof(u32)))
        goto end;
      totalCycles += 18 + 6 * transfer_size;
      u32 xf_address = Cmd2 & 0xFFFF;
      LoadXFReg(transfer_size, xf_address, DataReader(start_ptr, end_ptr));
      INCSTAT(stats.thisFrame.numXFLoads);
      start_ptr += 4 * (transfer_size);
    }
    break;

    case OpcodeDecoder::GX_LOAD_INDX_A:  // used for position matrices
      refarray = 0xC;
      goto load_indx;
    case OpcodeDecoder::GX_LOAD_INDX_B:  // used for normal matrices
      refarray = 0xD;
      goto load_indx;
    case OpcodeDecoder::GX_LOAD_INDX_C:  // used for postmatrices
      refarray = 0xE;
      goto load_indx;
    case OpcodeDecoder::GX_LOAD_INDX_D:  // used for lights
      refarray = 0xF;
      goto load_indx;
    load_indx:
      if (!FifoReserveMemory(start_ptr, end_ptr, 4))
        goto end;
      totalCycles += 6;
      LoadIndexedXF(Common::FromBigEndian<u32>(*((u32*)start_ptr)), refarray);
      start_ptr += 4;
      break;

    case OpcodeDecoder::GX_CMD_CALL_DL:
    {
      if (!FifoReserveMemory(start_ptr, end_ptr, 8))
        goto end;
      u32 address = Common::FromBigEndian<u32>(*((u32*)start_ptr));
      start_ptr += 4;
      u32 count = Common::FromBigEndian<u32>(*((u32*)start_ptr));
      start_ptr += 4;
      totalCycles += 6 + OpcodeDecoder::InterpretDisplayList(address, count);
    }
    break;

    case OpcodeDecoder::GX_CMD_UNKNOWN_METRICS:  // zelda 4 swords calls it and checks the metrics
                                                 // registers after that
      totalCycles += 6;
      DEBUG_LOG(VIDEO, "GX 0x44: %08x", cmd_byte);
      break;

    case OpcodeDecoder::GX_CMD_INVL_VC:  // Invalidate Vertex Cache
      totalCycles += 6;
      DEBUG_LOG(VIDEO, "Invalidate (vertex cache?)");
      break;

    case OpcodeDecoder::GX_LOAD_BP_REG:
      // In skipped_frame case: We have to let BP writes through because they set
      // tokens and stuff.  TODO: Call a much simplified LoadBPReg instead.
      {
        if (!FifoReserveMemory(start_ptr, end_ptr, 4))
          goto end;
        totalCycles += 12;
        u32 bp_cmd = Common::FromBigEndian<u32>(*((u32*)start_ptr));
        start_ptr += 4;
        LoadBPReg(bp_cmd);
        INCSTAT(stats.thisFrame.numBPLoads);
      }
      break;

    // draw primitives
    default:
      if ((cmd_byte & 0xC0) == 0x80)
      {
        // load vertices
        if (!FifoReserveMemory(start_ptr, end_ptr, 2))
          goto end;
        u16 num_vertices = Common::FromBigEndian<u16>(*((u16*)start_ptr));
        start_ptr += 2;
        if (num_vertices > 0)
        {
          int vtx_attr_group = cmd_byte & OpcodeDecoder::GX_VAT_MASK;  // Vertex loader index (0 - 7)
          int primitive = (cmd_byte & OpcodeDecoder::GX_PRIMITIVE_MASK) >> OpcodeDecoder::GX_PRIMITIVE_SHIFT;
          int bytes = VertexLoaderManager::GetVertexSize(vtx_attr_group) * num_vertices;
          if (!FifoReserveMemory(start_ptr, end_ptr, bytes))
            goto end;
          // draw vertices
          VertexLoaderManager::RunVertices(vtx_attr_group, primitive, num_vertices, DataReader(start_ptr, end_ptr));
          start_ptr += bytes;
          // 4 GPU ticks per vertex, 3 CPU ticks per GPU tick
          totalCycles += num_vertices * 4 * 3 + 6;
        }
      }
      else
      {
        CommandProcessor::HandleUnknownOpcode(cmd_byte, s_video_buffer_read_ptr, false);
        ERROR_LOG(VIDEO, "FIFO: Unknown Opcode(0x%02x @ %p, preprocessing = no)", cmd_byte, s_video_buffer_read_ptr);
        totalCycles += 1;
      }
      break;
    }

    FifoFlushMemory(start_ptr);
    if (cycles < totalCycles)
      break;
  }

end:
  cycles -= totalCycles;
  Common::AtomicStore(fifo.SafeCPReadPointer, fifo.CPReadPointer);
}

// Description: Main FIFO update loop
// Purpose: Keep the Core HW updated about the CPU-GPU distance
void RunGpuLoop()
{
  AsyncRequests::GetInstance()->SetEnable(true);
  AsyncRequests::GetInstance()->SetPassthrough(false);

  s_gpu_mainloop.Run(
      [] {
        const SConfig& param = SConfig::GetInstance();

        // Do nothing while paused
        if (!s_emu_running_state.IsSet())
          return;

        if (s_use_deterministic_gpu_thread)
        {
          AsyncRequests::GetInstance()->PullEvents();

          // All the fifo/CP stuff is on the CPU.  We just need to run the opcode decoder.
          u8* seen_ptr = s_video_buffer_seen_ptr;
          u8* write_ptr = s_video_buffer_write_ptr;
          // See comment in SyncGPU
          if (write_ptr > seen_ptr)
          {
            s_video_buffer_read_ptr =
                OpcodeDecoder::Run(DataReader(s_video_buffer_read_ptr, write_ptr), nullptr, false);
            s_video_buffer_seen_ptr = write_ptr;
          }
        }
        else
        {
          CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;

          AsyncRequests::GetInstance()->PullEvents();

          CommandProcessor::SetCPStatusFromGPU();

          // check if we are able to run this buffer
          while (!CommandProcessor::IsInterruptWaiting() && fifo.bFF_GPReadEnable &&
                 fifo.CPReadWriteDistance && !AtBreakpoint())
          {
            if (param.bSyncGPU && s_sync_ticks.load() < param.iSyncGpuMinDistance)
              break;

            int availableCycles = 100;
            OpcodeDecoderRun(availableCycles);
            int cyclesExecuted = 100 - availableCycles;

            CommandProcessor::SetCPStatusFromGPU();

            if (param.bSyncGPU)
            {
              cyclesExecuted = (int)(cyclesExecuted / param.fSyncGpuOverclock);
              int old = s_sync_ticks.fetch_sub(cyclesExecuted);
              if (old >= param.iSyncGpuMaxDistance &&
                  old - cyclesExecuted < param.iSyncGpuMaxDistance)
                s_sync_wakeup_event.Set();
            }

            // This call is pretty important in DualCore mode and must be called in the FIFO Loop.
            // If we don't, s_swapRequested or s_efbAccessRequested won't be set to false
            // leading the CPU thread to wait in Video_BeginField or Video_AccessEFB thus slowing
            // things down.
            AsyncRequests::GetInstance()->PullEvents();
          }

          // fast skip remaining GPU time if fifo is empty
          if (s_sync_ticks.load() > 0)
          {
            int old = s_sync_ticks.exchange(0);
            if (old >= param.iSyncGpuMaxDistance)
              s_sync_wakeup_event.Set();
          }

          // The fifo is empty and it's unlikely we will get any more work in the near future.
          // Make sure VertexManager finishes drawing any primitives it has stored in it's buffer.
          g_vertex_manager->Flush();
        }
      },
      100);

  AsyncRequests::GetInstance()->SetEnable(false);
  AsyncRequests::GetInstance()->SetPassthrough(true);
}

void FlushGpu()
{
  const SConfig& param = SConfig::GetInstance();

  if (!param.bCPUThread || s_use_deterministic_gpu_thread)
    return;

  s_gpu_mainloop.Wait();
}

void GpuMaySleep()
{
  s_gpu_mainloop.AllowSleep();
}

bool AtBreakpoint()
{
  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;
  return fifo.bFF_BPEnable && (fifo.CPReadPointer == fifo.CPBreakpoint);
}

void RunGpu()
{
  const SConfig& param = SConfig::GetInstance();

  // wake up GPU thread
  if (param.bCPUThread && !s_use_deterministic_gpu_thread)
  {
    s_gpu_mainloop.Wakeup();
  }

  // if the sync GPU callback is suspended, wake it up.
  if (!SConfig::GetInstance().bCPUThread || s_use_deterministic_gpu_thread ||
      SConfig::GetInstance().bSyncGPU)
  {
    if (s_syncing_suspended)
    {
      s_syncing_suspended = false;
      CoreTiming::ScheduleEvent(GPU_TIME_SLOT_SIZE, s_event_sync_gpu, GPU_TIME_SLOT_SIZE);
    }
  }
}

static int RunGpuOnCpu(int ticks)
{
  CommandProcessor::SCPFifoStruct& fifo = CommandProcessor::fifo;
  bool reset_simd_state = false;
  int available_ticks = int(ticks * SConfig::GetInstance().fSyncGpuOverclock) + s_sync_ticks.load();
  while (fifo.bFF_GPReadEnable && fifo.CPReadWriteDistance && !AtBreakpoint() &&
         available_ticks >= 0)
  {
    if (s_use_deterministic_gpu_thread)
    {
      u32 step_size = ReadDataFromFifoOnCPU(fifo.CPReadPointer);
      s_gpu_mainloop.Wakeup();

      if (fifo.CPReadPointer == fifo.CPEnd)
        fifo.CPReadPointer = fifo.CPBase;
      else
        fifo.CPReadPointer += step_size;

      fifo.CPReadWriteDistance -= step_size;
    }
    else
    {
      if (!reset_simd_state)
      {
        FPURoundMode::SaveSIMDState();
        FPURoundMode::LoadDefaultSIMDState();
        reset_simd_state = true;
      }
      OpcodeDecoderRun(available_ticks);
    }
  }

  CommandProcessor::SetCPStatusFromGPU();

  if (reset_simd_state)
  {
    FPURoundMode::LoadSIMDState();
  }

  // Discard all available ticks as there is nothing to do any more.
  s_sync_ticks.store(std::min(available_ticks, 0));

  // If the GPU is idle, drop the handler.
  if (available_ticks >= 0)
    return -1;

  // Always wait at least for GPU_TIME_SLOT_SIZE cycles.
  return -available_ticks + GPU_TIME_SLOT_SIZE;
}

void UpdateWantDeterminism(bool want)
{
  // We are paused (or not running at all yet), so
  // it should be safe to change this.
  const SConfig& param = SConfig::GetInstance();
  bool gpu_thread = false;
  switch (param.m_GPUDeterminismMode)
  {
  case GPUDeterminismMode::Auto:
    gpu_thread = want;
    break;
  case GPUDeterminismMode::Disabled:
    gpu_thread = false;
    break;
  case GPUDeterminismMode::FakeCompletion:
    gpu_thread = true;
    break;
  }

  gpu_thread = gpu_thread && param.bCPUThread;

  if (s_use_deterministic_gpu_thread != gpu_thread)
  {
    s_use_deterministic_gpu_thread = gpu_thread;
    if (gpu_thread)
    {
      // These haven't been updated in non-deterministic mode.
      s_video_buffer_seen_ptr = s_video_buffer_pp_read_ptr = s_video_buffer_read_ptr;
      CopyPreprocessCPStateFromMain();
      VertexLoaderManager::MarkAllDirty();
    }
  }
}

bool UseDeterministicGPUThread()
{
  return s_use_deterministic_gpu_thread;
}

/* This function checks the emulated CPU - GPU distance and may wake up the GPU,
 * or block the CPU if required. It should be called by the CPU thread regularly.
 * @ticks The gone emulated CPU time.
 * @return A good time to call WaitForGpuThread() next.
 */
static int WaitForGpuThread(int ticks)
{
  const SConfig& param = SConfig::GetInstance();

  int old = s_sync_ticks.fetch_add(ticks);
  int now = old + ticks;

  // GPU is idle, so stop polling.
  if (old >= 0 && s_gpu_mainloop.IsDone())
    return -1;

  // Wakeup GPU
  if (old < param.iSyncGpuMinDistance && now >= param.iSyncGpuMinDistance)
    RunGpu();

  // If the GPU is still sleeping, wait for a longer time
  if (now < param.iSyncGpuMinDistance)
    return GPU_TIME_SLOT_SIZE + param.iSyncGpuMinDistance - now;

  // Wait for GPU
  if (now >= param.iSyncGpuMaxDistance)
    s_sync_wakeup_event.Wait();

  return GPU_TIME_SLOT_SIZE;
}

static void SyncGPUCallback(u64 ticks, s64 cyclesLate)
{
  ticks += cyclesLate;
  int next = -1;

  if (!SConfig::GetInstance().bCPUThread || s_use_deterministic_gpu_thread)
  {
    next = RunGpuOnCpu((int)ticks);
  }
  else if (SConfig::GetInstance().bSyncGPU)
  {
    next = WaitForGpuThread((int)ticks);
  }

  s_syncing_suspended = next < 0;
  if (!s_syncing_suspended)
    CoreTiming::ScheduleEvent(next, s_event_sync_gpu, next);
}

// Initialize GPU - CPU thread syncing, this gives us a deterministic way to start the GPU thread.
void Prepare()
{
  s_event_sync_gpu = CoreTiming::RegisterEvent("SyncGPUCallback", SyncGPUCallback);
  s_syncing_suspended = true;
}
}  // namespace Fifo

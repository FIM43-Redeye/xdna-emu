// SPDX-License-Identifier: MIT

#include <libvfio-user.h>

#include "xdna_emu.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  BAR0 = 0,
  BAR2 = 2,
  BAR4 = 4,
};

enum {
  BAR0_BASE = 0x03000000,
  BAR2_BASE = 0x03080000,
  BAR4_BASE = 0x030c0000,
  BAR0_SIZE = 0x00080000,
  BAR2_SIZE = 0x00040000,
  BAR4_SIZE = 0x00040000,
  PSP_WAIT_MODE = 0x03010034,
  PSP_NOTIFY = 0x03010090,
  SMU_NOTIFY = 0x03010094,
  PSP_STATUS_CMD = 0x030100a0,
  PSP_ARG0 = 0x030100a4,
  PSP_ARG1 = 0x030100a8,
  SMU_CMD = 0x030100ac,
  SMU_RESPONSE = 0x030100b0,
  SMU_ARG = 0x030100b4,
  PSP_ARG2 = 0x030100bc,
  PSP_READY = 1u << 31,
  PSP_VALIDATE = 1,
  PSP_START = 2,
  PSP_RELEASE_TMR = 3,
  SMU_POWER_ON = 3,
  SMU_POWER_OFF = 4,
  SMU_SET_MPNPU_CLOCK = 5,
  SMU_SET_H_CLOCK = 6,
  SMU_SET_SOFT_DPM = 7,
  SMU_SET_HARD_DPM = 8,
  SMU_RESULT_OK = 1,
  FIRMWARE_BOOT_BUDGET = 200000,
};

typedef struct {
  uint64_t base;
  uint64_t size;
  uint8_t *data;
} ActiveMap;

typedef struct {
  uint32_t psp_status_cmd;
  uint32_t psp_arg0;
  uint32_t psp_arg1;
  uint32_t psp_arg2;
  uint32_t psp_notify;
  uint32_t smu_cmd;
  uint32_t smu_response;
  uint32_t smu_arg;
  uint32_t smu_notify;
  uint32_t wait_mode;
} Controller;

typedef struct {
  XdnaEmuHandle *emu;
  ActiveMap *maps;
  size_t map_count;
  size_t map_capacity;
  Controller regs;
  bool firmware_validated;
  bool firmware_started;
  bool powered;
  bool fatal;
  uint32_t mpnpu_clock;
  uint32_t h_clock;
  uint32_t soft_dpm;
  uint32_t hard_dpm;
  char fatal_message[256];
} PhoenixFrontend;

static bool bar_device_address(unsigned int bar, uint64_t offset, size_t count,
                               uint32_t *address);
static bool frontend_init(PhoenixFrontend *frontend);
static void frontend_destroy(PhoenixFrontend *frontend);
static bool frontend_map(PhoenixFrontend *frontend, uint64_t base,
                         uint8_t *data, uint64_t size);
static bool frontend_unmap(PhoenixFrontend *frontend, uint64_t base,
                           uint64_t size);
static bool frontend_cold_reset(PhoenixFrontend *frontend);
static bool frontend_bar_access(PhoenixFrontend *frontend, unsigned int bar,
                                uint64_t offset, void *data, size_t count,
                                bool is_write);

static void controller_reset(PhoenixFrontend *frontend) {
  memset(&frontend->regs, 0, sizeof(frontend->regs));
  frontend->regs.psp_status_cmd = PSP_READY;
  frontend->firmware_validated = false;
  frontend->firmware_started = false;
  frontend->powered = false;
  frontend->mpnpu_clock = 0;
  frontend->h_clock = 0;
  frontend->soft_dpm = 0;
  frontend->hard_dpm = 0;
  frontend->fatal = false;
  frontend->fatal_message[0] = '\0';
}

static void latch_fatal(PhoenixFrontend *frontend, const char *format, ...) {
  va_list arguments;

  if (frontend->fatal) {
    return;
  }
  frontend->fatal = true;
  va_start(arguments, format);
  vsnprintf(frontend->fatal_message, sizeof(frontend->fatal_message), format,
            arguments);
  va_end(arguments);
}

static void latch_ffi_error(PhoenixFrontend *frontend, const char *operation) {
  char detail[192] = {0};

  xdna_emu_get_error(detail, sizeof(detail));
  latch_fatal(frontend, "%s: %s", operation,
              detail[0] == '\0' ? "unknown emulator error" : detail);
}

static bool bar_device_address(unsigned int bar, uint64_t offset, size_t count,
                               uint32_t *address) {
  uint32_t base;
  uint64_t size;

  if (address == NULL || count == 0) {
    return false;
  }
  switch (bar) {
  case BAR0:
    base = BAR0_BASE;
    size = BAR0_SIZE;
    break;
  case BAR2:
    base = BAR2_BASE;
    size = BAR2_SIZE;
    break;
  case BAR4:
    base = BAR4_BASE;
    size = BAR4_SIZE;
    break;
  default:
    return false;
  }
  if (offset >= size || count > size - offset) {
    return false;
  }
  *address = base + (uint32_t)offset;
  return true;
}

static bool frontend_init(PhoenixFrontend *frontend) {
  memset(frontend, 0, sizeof(*frontend));
  frontend->emu = xdna_emu_create();
  if (frontend->emu == NULL) {
    latch_fatal(frontend, "failed to create emulator handle");
    return false;
  }
  controller_reset(frontend);
  return true;
}

static void frontend_destroy(PhoenixFrontend *frontend) {
  xdna_emu_destroy(frontend->emu);
  free(frontend->maps);
  memset(frontend, 0, sizeof(*frontend));
}

static bool frontend_map(PhoenixFrontend *frontend, uint64_t base,
                         uint8_t *data, uint64_t size) {
  if (xdna_emu_map_host_memory(frontend->emu, base, data, size) !=
      XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "map host memory");
    return false;
  }

  if (frontend->map_count == frontend->map_capacity) {
    size_t capacity =
        frontend->map_capacity == 0 ? 4 : frontend->map_capacity * 2;
    if (capacity < frontend->map_capacity ||
        capacity > SIZE_MAX / sizeof(*frontend->maps)) {
      xdna_emu_unmap_host_memory(frontend->emu, base, size);
      latch_fatal(frontend, "active-map list is too large");
      return false;
    }
    ActiveMap *maps =
        realloc(frontend->maps, capacity * sizeof(*frontend->maps));
    if (maps == NULL) {
      xdna_emu_unmap_host_memory(frontend->emu, base, size);
      latch_fatal(frontend, "failed to grow active-map list");
      return false;
    }
    frontend->maps = maps;
    frontend->map_capacity = capacity;
  }
  frontend->maps[frontend->map_count++] =
      (ActiveMap){.base = base, .size = size, .data = data};
  return true;
}

static bool frontend_unmap(PhoenixFrontend *frontend, uint64_t base,
                           uint64_t size) {
  size_t index;

  for (index = 0; index < frontend->map_count; ++index) {
    if (frontend->maps[index].base == base &&
        frontend->maps[index].size == size) {
      break;
    }
  }
  if (index == frontend->map_count) {
    latch_fatal(frontend, "unmap does not match an active range");
    return false;
  }
  if (xdna_emu_unmap_host_memory(frontend->emu, base, size) !=
      XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "unmap host memory");
    return false;
  }
  memmove(&frontend->maps[index], &frontend->maps[index + 1],
          (frontend->map_count - index - 1) * sizeof(*frontend->maps));
  --frontend->map_count;
  return true;
}

static bool frontend_cold_reset(PhoenixFrontend *frontend) {
  XdnaEmuHandle *replacement = xdna_emu_create();

  if (replacement == NULL) {
    latch_fatal(frontend, "failed to create replacement emulator handle");
    return false;
  }
  for (size_t index = 0; index < frontend->map_count; ++index) {
    ActiveMap *map = &frontend->maps[index];
    if (xdna_emu_map_host_memory(replacement, map->base, map->data,
                                 map->size) != XDNA_EMU_SUCCESS) {
      xdna_emu_destroy(replacement);
      latch_ffi_error(frontend, "replay host-memory mapping");
      return false;
    }
  }

  xdna_emu_destroy(frontend->emu);
  frontend->emu = replacement;
  controller_reset(frontend);
  return true;
}

static bool copy_guest(PhoenixFrontend *frontend, uint64_t address,
                       uint8_t *destination, uint64_t size) {
  uint64_t end;

  if (__builtin_add_overflow(address, size, &end)) {
    latch_fatal(frontend, "guest range overflows");
    return false;
  }
  while (address < end) {
    ActiveMap *found = NULL;
    for (size_t index = 0; index < frontend->map_count; ++index) {
      ActiveMap *map = &frontend->maps[index];
      if (address >= map->base && address < map->base + map->size) {
        found = map;
        break;
      }
    }
    if (found == NULL) {
      latch_fatal(frontend, "guest range is not fully mapped");
      return false;
    }
    uint64_t offset = address - found->base;
    uint64_t chunk = found->size - offset;
    if (chunk > end - address) {
      chunk = end - address;
    }
    memcpy(destination, found->data + offset, (size_t)chunk);
    destination += chunk;
    address += chunk;
  }
  return true;
}

static void complete_psp(PhoenixFrontend *frontend) {
  frontend->regs.psp_arg0 = 0;
  frontend->regs.psp_status_cmd = PSP_READY;
}

static void process_psp(PhoenixFrontend *frontend) {
  uint32_t command = frontend->regs.psp_status_cmd;

  switch (command) {
  case PSP_VALIDATE: {
    uint64_t address = (uint64_t)frontend->regs.psp_arg0 |
                       (uint64_t)frontend->regs.psp_arg1 << 32;
    uint32_t size = frontend->regs.psp_arg2 & 0x00ffffff;
    if (frontend->firmware_validated || frontend->firmware_started ||
        size == 0) {
      latch_fatal(frontend, "invalid PSP VALIDATE state or size");
      return;
    }
    uint8_t *firmware = malloc(size);
    if (firmware == NULL) {
      latch_fatal(frontend, "failed to allocate PSP firmware copy");
      return;
    }
    bool copied = copy_guest(frontend, address, firmware, size);
    if (copied && xdna_emu_load_firmware(frontend->emu, firmware, size) !=
                      XDNA_EMU_SUCCESS) {
      latch_ffi_error(frontend, "PSP VALIDATE");
      copied = false;
    }
    free(firmware);
    if (!copied) {
      return;
    }
    frontend->firmware_validated = true;
    complete_psp(frontend);
    return;
  }
  case PSP_START:
    if (!frontend->firmware_validated || frontend->firmware_started ||
        frontend->regs.psp_arg0 != 1 || frontend->regs.psp_arg1 != 0 ||
        frontend->regs.psp_arg2 != 0) {
      latch_fatal(frontend, "invalid PSP START state or arguments");
      return;
    }
    if (xdna_emu_boot_firmware(frontend->emu, FIRMWARE_BOOT_BUDGET) !=
        XDNA_EMU_SUCCESS) {
      latch_ffi_error(frontend, "PSP START");
      return;
    }
    frontend->firmware_started = true;
    complete_psp(frontend);
    return;
  case PSP_RELEASE_TMR:
    if (!frontend->firmware_started) {
      latch_fatal(frontend, "PSP RELEASE_TMR before START");
      return;
    }
    frontend_cold_reset(frontend);
    return;
  default:
    latch_fatal(frontend, "unsupported PSP command %u", command);
    return;
  }
}

static void process_smu(PhoenixFrontend *frontend) {
  switch (frontend->regs.smu_cmd) {
  case SMU_POWER_ON:
    frontend->powered = true;
    break;
  case SMU_POWER_OFF:
    frontend->powered = false;
    break;
  case SMU_SET_MPNPU_CLOCK:
    frontend->mpnpu_clock = frontend->regs.smu_arg;
    break;
  case SMU_SET_H_CLOCK:
    frontend->h_clock = frontend->regs.smu_arg;
    break;
  case SMU_SET_SOFT_DPM:
    frontend->soft_dpm = frontend->regs.smu_arg;
    break;
  case SMU_SET_HARD_DPM:
    frontend->hard_dpm = frontend->regs.smu_arg;
    break;
  default:
    latch_fatal(frontend, "unsupported SMU command %u", frontend->regs.smu_cmd);
    return;
  }
  frontend->regs.smu_response = SMU_RESULT_OK;
}

static bool controller_read32(PhoenixFrontend *frontend, uint32_t address,
                              uint32_t *value) {
  switch (address) {
  case PSP_WAIT_MODE:
    *value = frontend->regs.wait_mode;
    return true;
  case PSP_NOTIFY:
    *value = frontend->regs.psp_notify;
    return true;
  case SMU_NOTIFY:
    *value = frontend->regs.smu_notify;
    return true;
  case PSP_STATUS_CMD:
    *value = frontend->regs.psp_status_cmd;
    return true;
  case PSP_ARG0:
    *value = frontend->regs.psp_arg0;
    return true;
  case PSP_ARG1:
    *value = frontend->regs.psp_arg1;
    return true;
  case PSP_ARG2:
    *value = frontend->regs.psp_arg2;
    return true;
  case SMU_CMD:
    *value = frontend->regs.smu_cmd;
    return true;
  case SMU_RESPONSE:
    *value = frontend->regs.smu_response;
    return true;
  case SMU_ARG:
    *value = frontend->regs.smu_arg;
    return true;
  default:
    latch_fatal(frontend, "unsupported BAR0 read at %#x", address);
    return false;
  }
}

static bool controller_write32(PhoenixFrontend *frontend, uint32_t address,
                               uint32_t value) {
  switch (address) {
  case PSP_NOTIFY: {
    uint32_t previous = frontend->regs.psp_notify;
    frontend->regs.psp_notify = value;
    if (previous == 0 && value == 1) {
      process_psp(frontend);
    }
    return true;
  }
  case SMU_NOTIFY: {
    uint32_t previous = frontend->regs.smu_notify;
    frontend->regs.smu_notify = value;
    if (previous == 0 && value == 1) {
      process_smu(frontend);
    }
    return true;
  }
  case PSP_STATUS_CMD:
    frontend->regs.psp_status_cmd = value;
    return true;
  case PSP_ARG0:
    frontend->regs.psp_arg0 = value;
    return true;
  case PSP_ARG1:
    frontend->regs.psp_arg1 = value;
    return true;
  case PSP_ARG2:
    frontend->regs.psp_arg2 = value;
    return true;
  case SMU_CMD:
    frontend->regs.smu_cmd = value;
    return true;
  case SMU_ARG:
    frontend->regs.smu_arg = value;
    return true;
  default:
    latch_fatal(frontend, "unsupported BAR0 write at %#x", address);
    return false;
  }
}

static bool device_read32(PhoenixFrontend *frontend, unsigned int bar,
                          uint32_t address, uint32_t *value) {
  if (bar == BAR0) {
    return controller_read32(frontend, address, value);
  }
  if (xdna_emu_firmware_read_host32(frontend->emu, address, value) !=
      XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "BAR read");
    return false;
  }
  return true;
}

static bool device_write32(PhoenixFrontend *frontend, unsigned int bar,
                           uint32_t address, uint32_t value) {
  if (bar == BAR0) {
    return controller_write32(frontend, address, value);
  }
  if (xdna_emu_firmware_write_host32(frontend->emu, address, value) !=
      XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "BAR write");
    return false;
  }
  return true;
}

static bool frontend_bar_access(PhoenixFrontend *frontend, unsigned int bar,
                                uint64_t offset, void *data, size_t count,
                                bool is_write) {
  uint32_t address;
  uint8_t *bytes = data;
  size_t done = 0;

  if (frontend == NULL || frontend->emu == NULL || data == NULL ||
      frontend->fatal || !bar_device_address(bar, offset, count, &address)) {
    return false;
  }
  while (done < count) {
    uint32_t current = address + (uint32_t)done;
    uint32_t word_address = current & ~3u;
    size_t lane = current & 3u;
    size_t chunk = 4 - lane;
    uint32_t word = 0;
    if (chunk > count - done) {
      chunk = count - done;
    }

    if (!is_write || lane != 0 || chunk != 4) {
      if (!device_read32(frontend, bar, word_address, &word)) {
        return false;
      }
    }
    if (is_write) {
      for (size_t index = 0; index < chunk; ++index) {
        uint32_t shift = (uint32_t)(lane + index) * 8;
        word = (word & ~(0xffu << shift)) | (uint32_t)bytes[done + index]
                                                << shift;
      }
      if (!device_write32(frontend, bar, word_address, word)) {
        return false;
      }
    } else {
      for (size_t index = 0; index < chunk; ++index) {
        bytes[done + index] = (uint8_t)(word >> ((lane + index) * 8));
      }
    }
    done += chunk;
  }
  return true;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "self-test failed at %s:%d: %s\n", __FILE__, __LINE__,   \
              #condition);                                                     \
      return false;                                                            \
    }                                                                          \
  } while (0)

static void store_le32(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)value;
  bytes[1] = (uint8_t)(value >> 8);
  bytes[2] = (uint8_t)(value >> 16);
  bytes[3] = (uint8_t)(value >> 24);
}

static bool bar_write32(PhoenixFrontend *frontend, unsigned int bar,
                        uint32_t device_address, uint32_t value) {
  uint32_t base = bar == BAR0 ? BAR0_BASE : bar == BAR2 ? BAR2_BASE : BAR4_BASE;
  uint8_t bytes[4];
  store_le32(bytes, value);
  return frontend_bar_access(frontend, bar, device_address - base, bytes,
                             sizeof(bytes), true);
}

static bool bar_read32(PhoenixFrontend *frontend, unsigned int bar,
                       uint32_t device_address, uint32_t *value) {
  uint32_t base = bar == BAR0 ? BAR0_BASE : bar == BAR2 ? BAR2_BASE : BAR4_BASE;
  uint8_t bytes[4];
  if (!frontend_bar_access(frontend, bar, device_address - base, bytes,
                           sizeof(bytes), false)) {
    return false;
  }
  *value = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
  return true;
}

static bool notify(PhoenixFrontend *frontend, uint32_t address) {
  return bar_write32(frontend, BAR0, address, 0) &&
         bar_write32(frontend, BAR0, address, 1);
}

static bool issue_smu(PhoenixFrontend *frontend, uint32_t command,
                      uint32_t argument) {
  return bar_write32(frontend, BAR0, SMU_NOTIFY, 0) &&
         bar_write32(frontend, BAR0, SMU_CMD, command) &&
         bar_write32(frontend, BAR0, SMU_ARG, argument) &&
         bar_write32(frontend, BAR0, SMU_NOTIFY, 1);
}

static bool self_test(void) {
  enum { FIRMWARE_SIZE = 0x30000 };
  const uint64_t firmware_gpa = 0x60010000;
  uint32_t address = 0;
  PhoenixFrontend frontend;

  CHECK(bar_device_address(BAR0, 0x10034, 4, &address));
  CHECK(address == PSP_WAIT_MODE);
  CHECK(bar_device_address(BAR2, 0x1234, 8, &address));
  CHECK(address == BAR2_BASE + 0x1234);
  CHECK(bar_device_address(BAR4, BAR4_SIZE - 1, 1, &address));
  CHECK(address == BAR4_BASE + BAR4_SIZE - 1);
  CHECK(!bar_device_address(1, 0, 4, &address));
  CHECK(!bar_device_address(BAR0, BAR0_SIZE - 1, 2, &address));
  CHECK(!bar_device_address(BAR2, UINT64_MAX, 1, &address));
  CHECK(!bar_device_address(BAR4, 0, 0, &address));

  CHECK(frontend_init(&frontend));
  CHECK(frontend.regs.psp_status_cmd == PSP_READY);

  uint8_t *firmware = calloc(1, FIRMWARE_SIZE);
  CHECK(firmware != NULL);
  CHECK(frontend_map(&frontend, firmware_gpa, firmware, FIRMWARE_SIZE));

  /* Mutate after registration: PSP validation must read the live mapping. */
  memcpy(firmware + 0x10, "$PS1", 4);
  store_le32(firmware + 0x14, FIRMWARE_SIZE - 0x100);
  memcpy(firmware + 0x200, (uint8_t[]){0x00, 0x70, 0x00}, 3);

  CHECK(bar_write32(&frontend, BAR0, PSP_ARG0, (uint32_t)firmware_gpa));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG1, (uint32_t)(firmware_gpa >> 32)));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG2, FIRMWARE_SIZE));
  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_VALIDATE));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.firmware_validated);
  CHECK(!frontend.firmware_started);
  CHECK(frontend.regs.psp_status_cmd == PSP_READY);

  /* Notify remains high, so this is deliberately not another edge. */
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG0, 1));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG1, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG2, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_START));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(!frontend.firmware_started);
  CHECK(notify(&frontend, PSP_NOTIFY));
  CHECK(frontend.firmware_started);

  uint8_t initial[8] = {0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87};
  uint8_t actual[8] = {0};
  CHECK(frontend_bar_access(&frontend, BAR4, 0x200, initial, sizeof(initial),
                            true));
  CHECK(frontend_bar_access(&frontend, BAR4, 0x200, actual, sizeof(actual),
                            false));
  CHECK(memcmp(actual, initial, sizeof(actual)) == 0);

  uint8_t split[6] = {0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6};
  CHECK(
      frontend_bar_access(&frontend, BAR4, 0x201, split, sizeof(split), true));
  uint8_t one = 0x5a;
  uint8_t two[2] = {0x6b, 0x7c};
  CHECK(frontend_bar_access(&frontend, BAR4, 0x203, &one, 1, true));
  CHECK(frontend_bar_access(&frontend, BAR4, 0x205, two, 2, true));
  CHECK(frontend_bar_access(&frontend, BAR4, 0x200, actual, sizeof(actual),
                            false));
  CHECK(memcmp(actual,
               (uint8_t[]){0x10, 0xa1, 0xb2, 0x5a, 0xd4, 0x6b, 0x7c, 0x87},
               sizeof(actual)) == 0);

  CHECK(issue_smu(&frontend, SMU_POWER_ON, 0));
  CHECK(frontend.powered);
  CHECK(bar_write32(&frontend, BAR0, SMU_CMD, SMU_POWER_OFF));
  CHECK(bar_write32(&frontend, BAR0, SMU_NOTIFY, 1));
  CHECK(frontend.powered);
  CHECK(issue_smu(&frontend, SMU_POWER_OFF, 0));
  CHECK(!frontend.powered);
  CHECK(issue_smu(&frontend, SMU_SET_MPNPU_CLOCK, 600));
  CHECK(frontend.mpnpu_clock == 600 && frontend.regs.smu_arg == 600);
  CHECK(issue_smu(&frontend, SMU_SET_H_CLOCK, 800));
  CHECK(frontend.h_clock == 800 && frontend.regs.smu_arg == 800);
  CHECK(issue_smu(&frontend, SMU_SET_SOFT_DPM, 2));
  CHECK(frontend.soft_dpm == 2);
  CHECK(issue_smu(&frontend, SMU_SET_HARD_DPM, 3));
  CHECK(frontend.hard_dpm == 3);
  CHECK(frontend.regs.smu_response == SMU_RESULT_OK);

  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_RELEASE_TMR));
  CHECK(notify(&frontend, PSP_NOTIFY));
  CHECK(!frontend.firmware_validated && !frontend.firmware_started);
  CHECK(frontend.regs.psp_status_cmd == PSP_READY);
  CHECK(frontend.regs.psp_notify == 0);
  CHECK(!frontend.powered);

  uint8_t replay_probe = 0xa5;
  CHECK(xdna_emu_write_host_memory(frontend.emu, firmware_gpa + 0x80,
                                   &replay_probe, 1) == XDNA_EMU_SUCCESS);
  CHECK(firmware[0x80] == replay_probe);

  CHECK(bar_write32(&frontend, BAR0, PSP_ARG0, 1));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG1, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG2, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_START));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));
  CHECK(!frontend.fatal);
  CHECK(frontend.regs.psp_status_cmd == PSP_READY);
  CHECK(frontend.mpnpu_clock == 0 && frontend.h_clock == 0);

  CHECK(bar_write32(&frontend, BAR0, PSP_ARG0, 0x65000000));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG1, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG2, FIRMWARE_SIZE));
  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_VALIDATE));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));

  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, 4));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));

  CHECK(issue_smu(&frontend, 9, 0));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));

  CHECK(bar_read32(&frontend, BAR0, PSP_STATUS_CMD, &address));
  CHECK(address == PSP_READY);
  CHECK(frontend_unmap(&frontend, firmware_gpa, FIRMWARE_SIZE));
  frontend_destroy(&frontend);
  free(firmware);
  return true;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
    if (!self_test()) {
      return EXIT_FAILURE;
    }
    puts("phoenix-vfio-user self-test: PASS");
    return EXIT_SUCCESS;
  }
  fprintf(stderr, "usage: %s --self-test\n", argv[0]);
  return EXIT_FAILURE;
}

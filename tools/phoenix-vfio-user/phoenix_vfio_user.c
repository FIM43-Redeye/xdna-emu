// SPDX-License-Identifier: MIT

#include <libvfio-user.h>

#include "xdna_emu.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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
  BAR1_SIZE = 0x00002000,
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
  MAP_SMOKE_CARVEOUT_BASE = 0x60000000,
  MAP_SMOKE_CARVEOUT_SIZE = 0x10000000,
  MAP_SMOKE_GUEST_NONCE_ADDRESS = 0x60001000,
  MAP_SMOKE_SERVER_NONCE_ADDRESS = 0x60001008,
};

static const uint8_t MAP_SMOKE_GUEST_NONCE[] = {0x55, 0x50, 0x4e, 0x54,
                                                0x53, 0x45, 0x55, 0x47};
static const uint8_t MAP_SMOKE_SERVER_NONCE[] = {0x50, 0x4e, 0x52, 0x45,
                                                 0x56, 0x52, 0x45, 0x53};

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
  bool enabled;
  bool carveout_mapped;
  bool guest_nonce_seen;
  bool server_nonce_written;
  size_t registered;
  size_t unregistered;
} MapSmoke;

typedef struct {
  XdnaEmuHandle *emu;
  vfu_ctx_t *vfu;
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
  MapSmoke map_smoke;
  char fatal_message[256];
} PhoenixFrontend;

typedef struct {
  unsigned int bar;
  int region;
  size_t size;
  int flags;
} BarSpec;

static const BarSpec PHOENIX_BARS[] = {
    {BAR0, VFU_PCI_DEV_BAR0_REGION_IDX, BAR0_SIZE,
     VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM},
    {1, VFU_PCI_DEV_BAR1_REGION_IDX, BAR1_SIZE,
     VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM},
    {BAR2, VFU_PCI_DEV_BAR2_REGION_IDX, BAR2_SIZE,
     VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM | VFU_REGION_FLAG_64_BITS |
         VFU_REGION_FLAG_PREFETCH},
    {BAR4, VFU_PCI_DEV_BAR4_REGION_IDX, BAR4_SIZE,
     VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM},
};

static bool bar_device_address(unsigned int bar, uint64_t offset, size_t count,
                               uint32_t *address);
static bool frontend_init(PhoenixFrontend *frontend);
static void frontend_destroy(PhoenixFrontend *frontend);
static bool frontend_map(PhoenixFrontend *frontend, uint64_t base,
                         uint8_t *data, uint64_t size);
static bool frontend_unmap(PhoenixFrontend *frontend, uint64_t base,
                           uint64_t size);
static bool frontend_range_is_mapped(const PhoenixFrontend *frontend,
                                     uint64_t base, uint64_t size);
static bool frontend_cold_reset(PhoenixFrontend *frontend);
static bool frontend_bar_access(PhoenixFrontend *frontend, unsigned int bar,
                                uint64_t offset, void *data, size_t count,
                                bool is_write);
static bool frontend_setup_vfio(PhoenixFrontend *frontend, const char *path);
static bool map_smoke_progress(PhoenixFrontend *frontend);
static bool map_smoke_finish(PhoenixFrontend *frontend);
static void dma_register_cb(vfu_ctx_t *vfu, vfu_dma_info_t *info);
static void dma_unregister_cb(vfu_ctx_t *vfu, vfu_dma_info_t *info);
static int device_reset_cb(vfu_ctx_t *vfu, vfu_reset_type_t type);
static bool frontend_trigger_mask(PhoenixFrontend *frontend, uint32_t mask,
                                  int (*trigger)(vfu_ctx_t *, uint32_t));

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

static void latch_errno(PhoenixFrontend *frontend, const char *operation) {
  latch_fatal(frontend, "%s: %s", operation, strerror(errno));
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
  if (frontend->vfu != NULL) {
    vfu_setup_device_reset_cb(frontend->vfu, NULL);
    vfu_destroy_ctx(frontend->vfu);
  }
  xdna_emu_destroy(frontend->emu);
  free(frontend->maps);
  memset(frontend, 0, sizeof(*frontend));
}

static bool frontend_track_range(PhoenixFrontend *frontend, uint64_t base,
                                 uint8_t *data, uint64_t size) {
  uint64_t end;

  if (size == 0 || __builtin_add_overflow(base, size, &end)) {
    latch_fatal(frontend, "invalid DMA range");
    return false;
  }
  for (size_t index = 0; index < frontend->map_count; ++index) {
    const ActiveMap *map = &frontend->maps[index];
    if (base < map->base + map->size && map->base < end) {
      latch_fatal(frontend, "DMA range overlaps an active range");
      return false;
    }
  }
  if (frontend->map_count == frontend->map_capacity) {
    size_t capacity =
        frontend->map_capacity == 0 ? 4 : frontend->map_capacity * 2;
    if (capacity < frontend->map_capacity ||
        capacity > SIZE_MAX / sizeof(*frontend->maps)) {
      latch_fatal(frontend, "active DMA range list is too large");
      return false;
    }
    ActiveMap *maps =
        realloc(frontend->maps, capacity * sizeof(*frontend->maps));
    if (maps == NULL) {
      latch_fatal(frontend, "failed to grow active DMA range list");
      return false;
    }
    frontend->maps = maps;
    frontend->map_capacity = capacity;
  }
  frontend->maps[frontend->map_count++] =
      (ActiveMap){.base = base, .size = size, .data = data};
  return true;
}

static bool frontend_map(PhoenixFrontend *frontend, uint64_t base,
                         uint8_t *data, uint64_t size) {
  if (xdna_emu_map_host_memory(frontend->emu, base, data, size) !=
      XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "map host memory");
    return false;
  }
  if (!frontend_track_range(frontend, base, data, size)) {
    xdna_emu_unmap_host_memory(frontend->emu, base, size);
    return false;
  }
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
  if (frontend->maps[index].data != NULL &&
      xdna_emu_unmap_host_memory(frontend->emu, base, size) !=
          XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "unmap host memory");
    return false;
  }
  memmove(&frontend->maps[index], &frontend->maps[index + 1],
          (frontend->map_count - index - 1) * sizeof(*frontend->maps));
  --frontend->map_count;
  return true;
}

static bool frontend_range_is_mapped(const PhoenixFrontend *frontend,
                                     uint64_t base, uint64_t size) {
  uint64_t end;

  if (size == 0 || __builtin_add_overflow(base, size, &end)) {
    return false;
  }
  while (base < end) {
    uint64_t next = base;
    for (size_t index = 0; index < frontend->map_count; ++index) {
      const ActiveMap *map = &frontend->maps[index];
      uint64_t map_end = map->base + map->size;
      if (map->data != NULL && map->base <= base && map_end > next) {
        next = map_end;
      }
    }
    if (next == base) {
      return false;
    }
    base = next;
  }
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
    if (map->data == NULL) {
      continue;
    }
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
      if (map->data != NULL && address >= map->base &&
          address < map->base + map->size) {
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
  case SMU_RESPONSE:
    frontend->regs.smu_response = value;
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

static ssize_t region_access(PhoenixFrontend *frontend, unsigned int bar,
                             char *buffer, size_t count, loff_t offset,
                             bool is_write) {
  if (offset < 0 || !frontend_bar_access(frontend, bar, (uint64_t)offset,
                                         buffer, count, is_write)) {
    errno = frontend->fatal ? EIO : EINVAL;
    return -1;
  }
  return (ssize_t)count;
}

static ssize_t bar0_access_cb(vfu_ctx_t *vfu, char *buffer, size_t count,
                              loff_t offset, bool is_write) {
  return region_access(vfu_get_private(vfu), BAR0, buffer, count, offset,
                       is_write);
}

static ssize_t bar2_access_cb(vfu_ctx_t *vfu, char *buffer, size_t count,
                              loff_t offset, bool is_write) {
  return region_access(vfu_get_private(vfu), BAR2, buffer, count, offset,
                       is_write);
}

static ssize_t bar4_access_cb(vfu_ctx_t *vfu, char *buffer, size_t count,
                              loff_t offset, bool is_write) {
  return region_access(vfu_get_private(vfu), BAR4, buffer, count, offset,
                       is_write);
}

static vfu_region_access_cb_t *bar_callback(unsigned int bar) {
  switch (bar) {
  case BAR0:
    return bar0_access_cb;
  case BAR2:
    return bar2_access_cb;
  case BAR4:
    return bar4_access_cb;
  default:
    return NULL;
  }
}

static bool map_smoke_progress(PhoenixFrontend *frontend) {
  uint8_t guest_nonce[sizeof(MAP_SMOKE_GUEST_NONCE)];

  if (!frontend->map_smoke.enabled ||
      frontend->map_smoke.server_nonce_written) {
    return true;
  }
  if (!frontend_range_is_mapped(frontend, MAP_SMOKE_CARVEOUT_BASE,
                                MAP_SMOKE_CARVEOUT_SIZE)) {
    return true;
  }
  frontend->map_smoke.carveout_mapped = true;
  if (xdna_emu_read_host_memory(frontend->emu, MAP_SMOKE_GUEST_NONCE_ADDRESS,
                                guest_nonce,
                                sizeof(guest_nonce)) != XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "read map-smoke guest nonce");
    return false;
  }
  if (memcmp(guest_nonce, MAP_SMOKE_GUEST_NONCE, sizeof(guest_nonce)) != 0) {
    return true;
  }
  frontend->map_smoke.guest_nonce_seen = true;
  if (xdna_emu_write_host_memory(
          frontend->emu, MAP_SMOKE_SERVER_NONCE_ADDRESS, MAP_SMOKE_SERVER_NONCE,
          sizeof(MAP_SMOKE_SERVER_NONCE)) != XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "write map-smoke server nonce");
    return false;
  }
  frontend->map_smoke.server_nonce_written = true;
  puts("map-smoke: guest nonce observed; server nonce published");
  fflush(stdout);
  return true;
}

static bool map_smoke_finish(PhoenixFrontend *frontend) {
  if (!frontend->map_smoke.carveout_mapped) {
    latch_fatal(frontend, "QEMU did not map the full carveout GPA range");
  } else if (!frontend->map_smoke.guest_nonce_seen) {
    latch_fatal(frontend, "QEMU GPA mapping did not expose the guest nonce");
  } else if (!frontend->map_smoke.server_nonce_written) {
    latch_fatal(frontend, "server nonce was not written through the GPA map");
  } else if (frontend->map_count != 0 || frontend->map_smoke.registered !=
                                             frontend->map_smoke.unregistered) {
    latch_fatal(frontend, "QEMU did not exactly unmap every DMA range");
  }
  if (frontend->fatal) {
    return false;
  }
  printf("map-smoke: PASS (%zu exact DMA map/unmap pairs)\n",
         frontend->map_smoke.registered);
  fflush(stdout);
  return true;
}

static bool frontend_dma_register(PhoenixFrontend *frontend,
                                  vfu_dma_info_t *info) {
  const uint32_t required_prot = PROT_READ | PROT_WRITE;
  uint64_t base;
  bool direct;
  bool direct_rw;
  bool indirect;
  bool read_only;

  if (info == NULL) {
    latch_fatal(frontend, "DMA registration had no metadata");
    return false;
  }
  base = (uint64_t)(uintptr_t)info->iova.iov_base;
  if (info->iova.iov_len == 0 || info->iova.iov_len > UINT64_MAX - base) {
    latch_fatal(frontend, "invalid DMA registration range");
    return false;
  }
  direct = info->vaddr != NULL && info->mapping.iov_base != NULL &&
           info->mapping.iov_len != 0;
  indirect = info->vaddr == NULL && info->mapping.iov_base == NULL &&
             info->mapping.iov_len == 0;
  direct_rw = direct && info->page_size != 0 &&
              (info->prot & required_prot) == required_prot;
  read_only =
      (direct || indirect) && info->page_size != 0 && info->prot == PROT_READ;
  if (!direct_rw && !read_only) {
    latch_fatal(frontend,
                "DMA range is neither direct RW nor a read-only overlay: "
                "GPA=%#llx size=%#zx "
                "vaddr=%p mapping=%p/%#zx page=%#zx prot=%#x",
                (unsigned long long)base, info->iova.iov_len, info->vaddr,
                info->mapping.iov_base, info->mapping.iov_len, info->page_size,
                info->prot);
    return false;
  }
  if (direct_rw
          ? !frontend_map(frontend, base, info->vaddr, info->iova.iov_len)
          : !frontend_track_range(frontend, base, NULL, info->iova.iov_len)) {
    return false;
  }
  if (frontend->map_smoke.enabled) {
    ++frontend->map_smoke.registered;
    printf("map-smoke: %s GPA=%#llx size=%#zx page=%#zx prot=%#x "
           "vaddr=%p\n",
           direct_rw ? "map" : "track", (unsigned long long)base,
           info->iova.iov_len, info->page_size, info->prot, info->vaddr);
    fflush(stdout);
  }
  return map_smoke_progress(frontend);
}

static void dma_register_cb(vfu_ctx_t *vfu, vfu_dma_info_t *info) {
  frontend_dma_register(vfu_get_private(vfu), info);
}

static void dma_unregister_cb(vfu_ctx_t *vfu, vfu_dma_info_t *info) {
  PhoenixFrontend *frontend = vfu_get_private(vfu);

  if (info == NULL) {
    latch_fatal(frontend, "invalid DMA unregister");
    return;
  }
  if (!frontend_unmap(frontend, (uint64_t)(uintptr_t)info->iova.iov_base,
                      info->iova.iov_len)) {
    if (!frontend->fatal) {
      latch_fatal(frontend, "invalid DMA unregister");
    }
    return;
  }
  if (frontend->map_smoke.enabled) {
    ++frontend->map_smoke.unregistered;
    printf("map-smoke: unmap GPA=%#llx size=%#zx\n",
           (unsigned long long)(uintptr_t)info->iova.iov_base,
           info->iova.iov_len);
    fflush(stdout);
  }
}

static int device_reset_cb(vfu_ctx_t *vfu, vfu_reset_type_t type) {
  PhoenixFrontend *frontend = vfu_get_private(vfu);

  switch (type) {
  case VFU_RESET_DEVICE:
  case VFU_RESET_LOST_CONN:
  case VFU_RESET_PCI_FLR:
    break;
  default:
    latch_fatal(frontend, "unsupported reset type %d", type);
    errno = EINVAL;
    return -1;
  }
  if (!frontend_cold_reset(frontend)) {
    errno = EIO;
    return -1;
  }
  return 0;
}

static bool frontend_trigger_mask(PhoenixFrontend *frontend, uint32_t mask,
                                  int (*trigger)(vfu_ctx_t *, uint32_t)) {
  if ((mask & ~0xffffu) != 0) {
    latch_fatal(frontend, "firmware returned an out-of-range MSI-X bit");
    return false;
  }
  for (uint32_t vector = 0; vector < 16; ++vector) {
    if ((mask & (1u << vector)) != 0 && trigger(frontend->vfu, vector) < 0) {
      latch_errno(frontend, "trigger MSI-X");
      return false;
    }
  }
  return true;
}

static bool frontend_setup_vfio(PhoenixFrontend *frontend, const char *path) {
  struct pxcap express = {
      .hdr.id = PCI_CAP_ID_EXP,
      .pxcaps = {.ver = 2, .dpt = PCI_EXP_TYPE_ENDPOINT},
      .pxdcap = {.flrc = 1},
  };
  struct msixcap msix = {
      .hdr.id = PCI_CAP_ID_MSIX,
      .mxc = {.ts = 15},
      .mtab = {.tbir = 1, .to = 0},
      .mpba = {.pbir = 1, .pbao = 0x1000 >> 3},
  };

  if (path == NULL || frontend->vfu != NULL) {
    latch_fatal(frontend, "invalid vfio-user setup");
    return false;
  }
  frontend->vfu =
      vfu_create_ctx(VFU_TRANS_SOCK, path, LIBVFIO_USER_FLAG_ATTACH_NB,
                     frontend, VFU_DEV_TYPE_PCI);
  if (frontend->vfu == NULL) {
    latch_errno(frontend, "create vfio-user context");
    return false;
  }
  if (vfu_pci_init(frontend->vfu, VFU_PCI_TYPE_EXPRESS, PCI_HEADER_TYPE_NORMAL,
                   0) < 0) {
    latch_errno(frontend, "initialize PCI function");
    return false;
  }
  vfu_pci_set_id(frontend->vfu, 0x1022, 0x1502, 0xf111, 0x0005);
  vfu_pci_set_class(frontend->vfu, 0x11, 0x80, 0);

  for (size_t index = 0; index < sizeof(PHOENIX_BARS) / sizeof(PHOENIX_BARS[0]);
       ++index) {
    const BarSpec *bar = &PHOENIX_BARS[index];
    if (vfu_setup_region(frontend->vfu, bar->region, bar->size,
                         bar_callback(bar->bar), bar->flags, NULL, 0, -1,
                         0) < 0) {
      latch_errno(frontend, "configure PCI BAR");
      return false;
    }
  }
  if (vfu_setup_device_dma(frontend->vfu, LIBVFIO_USER_MAX_DMA_REGIONS,
                           dma_register_cb, dma_unregister_cb) < 0 ||
      vfu_setup_device_nr_irqs(frontend->vfu, VFU_DEV_MSIX_IRQ, 16) < 0 ||
      vfu_setup_device_reset_cb(frontend->vfu, device_reset_cb) < 0 ||
      vfu_pci_add_capability(frontend->vfu, 0, 0, &express) < 0 ||
      vfu_pci_add_capability(frontend->vfu, 0, 0, &msix) < 0 ||
      vfu_realize_ctx(frontend->vfu) < 0) {
    latch_errno(frontend, "configure Phoenix PCI function");
    return false;
  }
  return true;
}

static bool frontend_service(PhoenixFrontend *frontend, bool *quiescent) {
  *quiescent = true;
  if (frontend->fatal) {
    return false;
  }
  if (!frontend->firmware_started) {
    return true;
  }

  XdnaEmuFirmwareServiceStatus status =
      xdna_emu_service_firmware(frontend->emu, 1, FIRMWARE_BOOT_BUDGET);
  if (status.result != XDNA_EMU_SUCCESS) {
    latch_ffi_error(frontend, "service firmware");
    return false;
  }
  bool wait_mode_changed = frontend->regs.wait_mode != (status.wait_mode != 0);
  frontend->regs.wait_mode = status.wait_mode != 0;
  *quiescent = status.quiescent != 0;
  if (status.pending_msix_mask != 0 || wait_mode_changed) {
    fprintf(stderr,
            "phoenix-vfio-user: firmware service msix=%#x wait_mode=%d "
            "quiescent=%d\n",
            status.pending_msix_mask, status.wait_mode, status.quiescent);
    fflush(stderr);
  }
  return frontend_trigger_mask(frontend, status.pending_msix_mask,
                               vfu_irq_trigger);
}

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number) {
  (void)signal_number;
  stop_requested = 1;
}

static bool frontend_run(PhoenixFrontend *frontend) {
  bool attached = false;

  while (!stop_requested && !frontend->fatal) {
    if (!attached) {
      if (vfu_attach_ctx(frontend->vfu) == 0) {
        attached = true;
      } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        latch_errno(frontend, "attach vfio-user client");
        break;
      }
    }

    int requests = 0;
    bool quiescent = true;
    if (attached) {
      requests = vfu_run_ctx(frontend->vfu);
      if (requests < 0) {
        if (errno == ENOTCONN) {
          if (frontend->map_smoke.enabled) {
            return map_smoke_finish(frontend);
          }
          attached = false;
          continue;
        }
        if (errno == EINTR) {
          continue;
        }
        latch_errno(frontend, "run vfio-user request");
        break;
      }
      if (!map_smoke_progress(frontend)) {
        break;
      }
      if (!frontend_service(frontend, &quiescent)) {
        break;
      }
      if (requests > 0 || !quiescent) {
        continue;
      }
    }

    struct pollfd pollfd = {
        .fd = vfu_get_poll_fd(frontend->vfu),
        .events = POLLIN,
    };
    if (pollfd.fd < 0) {
      latch_errno(frontend, "get vfio-user poll fd");
      break;
    }
    if (poll(&pollfd, 1, -1) < 0 && errno != EINTR) {
      latch_errno(frontend, "poll vfio-user socket");
      break;
    }
    if ((pollfd.revents & POLLNVAL) != 0) {
      latch_fatal(frontend, "vfio-user poll fd became invalid");
      break;
    }
  }
  return !frontend->fatal;
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
  return bar_write32(frontend, BAR0, SMU_RESPONSE, 0) &&
         bar_write32(frontend, BAR0, SMU_ARG, argument) &&
         bar_write32(frontend, BAR0, SMU_CMD, command) &&
         bar_write32(frontend, BAR0, SMU_NOTIFY, 0) &&
         bar_write32(frontend, BAR0, SMU_NOTIFY, 1);
}

static uint32_t test_irq_counts[16];
static int test_irq_failure = -1;

static int test_irq_trigger(vfu_ctx_t *vfu, uint32_t vector) {
  (void)vfu;
  if ((int)vector == test_irq_failure) {
    errno = EIO;
    return -1;
  }
  ++test_irq_counts[vector];
  return 0;
}

static void clear_expected_test_fatal(PhoenixFrontend *frontend) {
  frontend->fatal = false;
  frontend->fatal_message[0] = '\0';
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
  CHECK(frontend_range_is_mapped(&frontend, firmware_gpa, FIRMWARE_SIZE));
  CHECK(
      frontend_range_is_mapped(&frontend, firmware_gpa + 1, FIRMWARE_SIZE - 1));
  CHECK(!frontend_range_is_mapped(&frontend, firmware_gpa - 1,
                                  FIRMWARE_SIZE + 1));
  CHECK(!frontend_range_is_mapped(&frontend, firmware_gpa, 0));
  uint8_t adjacent[4] = {0};
  CHECK(frontend_map(&frontend, firmware_gpa + FIRMWARE_SIZE, adjacent,
                     sizeof(adjacent)));
  CHECK(
      frontend_range_is_mapped(&frontend, firmware_gpa + FIRMWARE_SIZE - 4, 8));
  CHECK(frontend_unmap(&frontend, firmware_gpa + FIRMWARE_SIZE,
                       sizeof(adjacent)));

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
  bool quiescent = false;
  CHECK(!frontend_service(&frontend, &quiescent));
  CHECK(quiescent);
  CHECK(frontend_cold_reset(&frontend));
  CHECK(frontend.fatal);
  clear_expected_test_fatal(&frontend);
  CHECK(frontend.regs.psp_status_cmd == PSP_READY);
  CHECK(frontend.mpnpu_clock == 0 && frontend.h_clock == 0);

  CHECK(bar_write32(&frontend, BAR0, PSP_ARG0, 0x65000000));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG1, 0));
  CHECK(bar_write32(&frontend, BAR0, PSP_ARG2, FIRMWARE_SIZE));
  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, PSP_VALIDATE));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  CHECK(bar_write32(&frontend, BAR0, PSP_STATUS_CMD, 4));
  CHECK(bar_write32(&frontend, BAR0, PSP_NOTIFY, 1));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  CHECK(issue_smu(&frontend, 9, 0));
  CHECK(frontend.fatal);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  char self_test_socket[96];
  snprintf(self_test_socket, sizeof(self_test_socket),
           "/tmp/xdna-emu-phoenix-vfio-%ld.sock", (long)getpid());
  if (!frontend_setup_vfio(&frontend, self_test_socket)) {
    fprintf(stderr, "vfio setup diagnostic: %s\n", frontend.fatal_message);
    CHECK(false);
  }
  CHECK(sizeof(PHOENIX_BARS) / sizeof(PHOENIX_BARS[0]) == 4);
  CHECK(PHOENIX_BARS[0].bar == BAR0 && PHOENIX_BARS[0].size == BAR0_SIZE);
  CHECK(PHOENIX_BARS[1].bar == 1 && PHOENIX_BARS[1].size == BAR1_SIZE);
  CHECK(PHOENIX_BARS[2].bar == BAR2 && PHOENIX_BARS[2].size == BAR2_SIZE);
  CHECK((PHOENIX_BARS[2].flags &
         (VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH)) ==
        (VFU_REGION_FLAG_64_BITS | VFU_REGION_FLAG_PREFETCH));
  CHECK(PHOENIX_BARS[3].bar == BAR4 && PHOENIX_BARS[3].size == BAR4_SIZE);

  vfu_pci_config_space_t *config = vfu_pci_get_config_space(frontend.vfu);
  CHECK(config->hdr.id.vid == 0x1022 && config->hdr.id.did == 0x1502);
  CHECK(config->hdr.ss.vid == 0xf111 && config->hdr.ss.sid == 0x0005);
  CHECK(config->hdr.rid == 0);
  CHECK(config->hdr.cc.bcc == 0x11 && config->hdr.cc.scc == 0x80 &&
        config->hdr.cc.pi == 0);
  CHECK(config->hdr.bars[0].mem.region_type == 0 &&
        config->hdr.bars[0].mem.prefetchable == 0);
  CHECK(config->hdr.bars[1].mem.region_type == 0 &&
        config->hdr.bars[1].mem.prefetchable == 0);
  CHECK(config->hdr.bars[2].mem.locatable ==
            PCI_BASE_ADDRESS_MEM_TYPE_LOCATABLE_64 &&
        config->hdr.bars[2].mem.prefetchable == 1);
  CHECK(config->hdr.bars[4].mem.region_type == 0 &&
        config->hdr.bars[4].mem.prefetchable == 0);

  size_t express_offset =
      vfu_pci_find_capability(frontend.vfu, false, PCI_CAP_ID_EXP);
  size_t msix_offset =
      vfu_pci_find_capability(frontend.vfu, false, PCI_CAP_ID_MSIX);
  CHECK(express_offset != 0 && msix_offset != 0);
  struct pxcap express;
  struct msixcap msix;
  memcpy(&express, &config->raw[express_offset], sizeof(express));
  memcpy(&msix, &config->raw[msix_offset], sizeof(msix));
  CHECK(express.pxcaps.ver == 2);
  CHECK(express.pxcaps.dpt == PCI_EXP_TYPE_ENDPOINT);
  CHECK(express.pxdcap.flrc == 1);
  CHECK(msix.mxc.ts == 15);
  CHECK(msix.mtab.tbir == 1 && msix.mtab.to == 0);
  CHECK(msix.mpba.pbir == 1 && msix.mpba.pbao == (0x1000 >> 3));
  CHECK(vfu_pci_find_capability(frontend.vfu, true, PCI_EXT_CAP_ID_PASID) == 0);

  uint8_t dma_bytes[0x1000] = {0};
  vfu_dma_info_t dma = {
      .iova = {.iov_base = (void *)(uintptr_t)0x60050000,
               .iov_len = sizeof(dma_bytes)},
      .vaddr = dma_bytes,
      .mapping = {.iov_base = dma_bytes, .iov_len = sizeof(dma_bytes)},
      .page_size = 0x1000,
      .prot = PROT_READ | PROT_WRITE,
  };
  dma_register_cb(frontend.vfu, &dma);
  CHECK(!frontend.fatal && frontend.map_count == 2);
  uint8_t dma_probe = 0xc7;
  CHECK(xdna_emu_write_host_memory(frontend.emu, 0x60050020, &dma_probe, 1) ==
        XDNA_EMU_SUCCESS);
  CHECK(dma_bytes[0x20] == dma_probe);

  vfu_dma_info_t rom = {
      .iova = {.iov_base = (void *)(uintptr_t)0xfffc0000, .iov_len = 0x40000},
      .vaddr = NULL,
      .mapping = {.iov_base = NULL, .iov_len = 0},
      .page_size = 0x1000,
      .prot = PROT_READ,
  };
  dma_register_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 3);
  CHECK(!frontend_range_is_mapped(&frontend, 0xfffc0000, 0x40000));
  CHECK(frontend_cold_reset(&frontend));
  CHECK(!frontend.fatal);
  CHECK(!frontend_range_is_mapped(&frontend, 0xfffc0000, 0x40000));
  dma_unregister_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 2);

  rom.iova.iov_base = (void *)(uintptr_t)0xc0000;
  rom.iova.iov_len = 0x20000;
  dma_register_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 3);
  CHECK(!frontend_range_is_mapped(&frontend, 0xc0000, 0x20000));
  dma_unregister_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 2);

  uint8_t direct_rom_bytes[0x1000] = {0};
  rom.iova.iov_base = (void *)(uintptr_t)0xc3000;
  rom.iova.iov_len = sizeof(direct_rom_bytes);
  rom.vaddr = direct_rom_bytes;
  rom.mapping.iov_base = direct_rom_bytes;
  rom.mapping.iov_len = sizeof(direct_rom_bytes);
  dma_register_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 3);
  CHECK(
      !frontend_range_is_mapped(&frontend, 0xc3000, sizeof(direct_rom_bytes)));
  CHECK(frontend_cold_reset(&frontend));
  CHECK(!frontend.fatal);
  CHECK(
      !frontend_range_is_mapped(&frontend, 0xc3000, sizeof(direct_rom_bytes)));
  dma_unregister_cb(frontend.vfu, &rom);
  CHECK(!frontend.fatal && frontend.map_count == 2);

  vfu_dma_info_t bad_dma = dma;
  bad_dma.iova.iov_base = (void *)(uintptr_t)0x60060000;
  bad_dma.vaddr = NULL;
  dma_register_cb(frontend.vfu, &bad_dma);
  CHECK(frontend.fatal && frontend.map_count == 2);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  bad_dma = dma;
  bad_dma.iova.iov_base = (void *)(uintptr_t)0x60060000;
  bad_dma.vaddr = NULL;
  bad_dma.mapping.iov_base = NULL;
  bad_dma.mapping.iov_len = 0;
  dma_register_cb(frontend.vfu, &bad_dma);
  CHECK(frontend.fatal && frontend.map_count == 2);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  bad_dma = dma;
  bad_dma.iova.iov_len /= 2;
  dma_unregister_cb(frontend.vfu, &bad_dma);
  CHECK(frontend.fatal && frontend.map_count == 2);
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);
  dma_unregister_cb(frontend.vfu, &dma);
  CHECK(!frontend.fatal && frontend.map_count == 1);

  memset(test_irq_counts, 0, sizeof(test_irq_counts));
  CHECK(frontend_trigger_mask(&frontend, (1u << 0) | (1u << 5) | (1u << 15),
                              test_irq_trigger));
  CHECK(test_irq_counts[0] == 1 && test_irq_counts[5] == 1 &&
        test_irq_counts[15] == 1);
  for (size_t vector = 0; vector < 16; ++vector) {
    if (vector != 0 && vector != 5 && vector != 15) {
      CHECK(test_irq_counts[vector] == 0);
    }
  }
  test_irq_failure = 7;
  CHECK(!frontend_trigger_mask(&frontend, 1u << 7, test_irq_trigger));
  CHECK(frontend.fatal);
  test_irq_failure = -1;
  CHECK(frontend_cold_reset(&frontend));
  clear_expected_test_fatal(&frontend);

  for (vfu_reset_type_t type = VFU_RESET_DEVICE; type <= VFU_RESET_PCI_FLR;
       type++) {
    CHECK(device_reset_cb(frontend.vfu, type) == 0);
    CHECK(frontend.regs.psp_status_cmd == PSP_READY);
    replay_probe = (uint8_t)(0xd0 + type);
    CHECK(xdna_emu_write_host_memory(frontend.emu, firmware_gpa + 0x80,
                                     &replay_probe, 1) == XDNA_EMU_SUCCESS);
    CHECK(firmware[0x80] == replay_probe);
  }

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
  bool map_smoke = argc == 3 && strcmp(argv[1], "--map-smoke") == 0;
  if (argc == 2 || map_smoke) {
    const char *path = argv[map_smoke ? 2 : 1];
    PhoenixFrontend frontend;
    if (!frontend_init(&frontend)) {
      fprintf(stderr, "phoenix-vfio-user: %s\n", frontend.fatal_message);
      frontend_destroy(&frontend);
      return EXIT_FAILURE;
    }
    frontend.map_smoke.enabled = map_smoke;
    if (!frontend_setup_vfio(&frontend, path)) {
      fprintf(stderr, "phoenix-vfio-user: %s\n", frontend.fatal_message);
      frontend_destroy(&frontend);
      return EXIT_FAILURE;
    }
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);
    printf("phoenix-vfio-user: listening on %s\n", path);
    fflush(stdout);
    bool success = frontend_run(&frontend);
    if (!success) {
      fprintf(stderr, "phoenix-vfio-user: %s\n", frontend.fatal_message);
    }
    frontend_destroy(&frontend);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  fprintf(stderr, "usage: %s --self-test | [--map-smoke] SOCKET_PATH\n",
          argv[0]);
  return EXIT_FAILURE;
}

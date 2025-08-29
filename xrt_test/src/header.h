#ifndef HEADER_H_GUARD
#define HEADER_H_GUARD

#include <stdint.h>

#include <experimental/xrt_bo.h>
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>

#define CONTROL_ADDRESS 0xA0000000

uint64_t perf_counter_ns();

xclDeviceHandle xclOpenDevice();

uint8_t get_done(void* base_ptr);
uint8_t get_idle(void* base_ptr);
uint8_t get_ready(void* base_ptr);
uint32_t get_32b(void* base_ptr, uint64_t offset);
void write_32b(void* base_ptr, uint64_t offset, uint32_t v);
void write_64b(void* base_ptr, uint64_t offset_l, uint64_t offset_h, uint64_t v);

void xclCloseDevice(xclDeviceHandle device);

xclBufferHandle xclAllocate(xclDeviceHandle device, uint32_t size);

void xclFree(xclDeviceHandle device, xclBufferHandle buffer);

char *xclMapBuffer(xclDeviceHandle device, xclBufferHandle buffer);

void xclUnmapBuffer(xclDeviceHandle device, xclBufferHandle buffer, char *buffer_ptr);

void xclFlush(xclDeviceHandle device, xclBufferHandle dest, uint32_t size);

void xclInvalidate(xclDeviceHandle device, xclBufferHandle src, uint32_t size);

void* open_device();

void *getBufferPhysicalAddress(xclDeviceHandle device, xclBufferHandle buffer);
#endif // HEADER_H_GUARD

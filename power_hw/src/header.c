#include "header.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <stdint.h>
#include <sys/sysinfo.h>


uint64_t perf_counter_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)(ts.tv_sec) * 1000000000 + (uint64_t)ts.tv_nsec;
}

/*
uint32_t orbitboost_done_or_idle(void *base_ptr)
{
    uint32_t state = 0x6;
    
    volatile uint32_t *ptr = base_ptr;

    state &= *((volatile uint32_t*)(ptr + XLDL_DSOLVE_CONTROL_ADDR_AP_CTRL)) & 0x6;
    return state;
}
*/

uint8_t get_done(void* base_ptr) {
    volatile uint32_t* ptr = base_ptr;
    return (*((volatile uint32_t*)(ptr + 0x00)) & 0x02) >> 1;
}

uint8_t get_idle(void* base_ptr) {
    volatile uint32_t* ptr = base_ptr;
    return (*((volatile uint32_t*)(ptr + 0x00)) & 0x04) >> 2;
}

uint8_t get_ready(void* base_ptr) {
    volatile uint32_t* ptr = base_ptr;
    return (*((volatile uint32_t*)(ptr + 0x00)) & 0x08) >> 3;
}

uint32_t get_32b(void* base_ptr, uint64_t offset) {
    volatile uint32_t* ptr = base_ptr;
    return *(ptr + offset);
}

void write_32b(void* base_ptr, uint64_t offset, uint32_t v) {
    volatile uint32_t* ptr = (volatile uint32_t*)(base_ptr + offset);
    //printf("writing %#08x at %#08x\n", v, offset); 
    *ptr = v;
}

void write_64b(void* base_ptr, uint64_t offset_l, uint64_t offset_h, uint64_t v) {
    uint32_t vl = (uint32_t)(v & 0xFFFFFFFF);
    uint32_t vh = (uint32_t)((v >> 32) & 0xFFFFFFFF);
    //printf("%#016x -> %#08x + %#08x\n", v, vh, vl);
    write_32b(base_ptr, offset_l, vl);
    write_32b(base_ptr, offset_h, vh);
}

xclDeviceHandle xclOpenDevice()
{
    xclDeviceHandle device_handle;
    device_handle = xclOpen(0, NULL, 0);

    if (!device_handle) {
        printf("ERROR: Failed to open device\n");
        return NULL;
    }

    return device_handle;
}

void xclCloseDevice(xclDeviceHandle device)
{
    xclClose(device);
}

xclBufferHandle xclAllocate(xclDeviceHandle device, uint32_t size)
{
    xclBufferHandle buffer;
    buffer = xclAllocBO(device, size, XCL_BO_DEVICE_RAM, 0);

    return buffer;
}

void xclFree(xclDeviceHandle device, xclBufferHandle buffer)
{
    if (buffer)
        xclFreeBO(device, buffer);
}



char *xclMapBuffer(xclDeviceHandle device, xclBufferHandle buffer)
{
    char *buffer_ptr;
    buffer_ptr = xclMapBO(device, buffer, true);

    if (buffer_ptr == NULL) {
        printf("ERROR: Failed to map BO buffer\n");
        exit(1);
    }

    return buffer_ptr;
}

void xclUnmapBuffer(xclDeviceHandle device, xclBufferHandle buffer, char *buffer_ptr)
{
    xclUnmapBO(device, buffer, buffer_ptr);
}

void xclFlush(xclDeviceHandle device, xclBufferHandle dest, uint32_t size)
{
    if (xclSyncBO(device, dest, XCL_BO_SYNC_BO_TO_DEVICE, size, 0)) {
        printf("ERROR: Failed to flush BO buffer\n");
        exit(1);
    }
}

void xclInvalidate(xclDeviceHandle device, xclBufferHandle src, uint32_t size)
{
    if (xclSyncBO(device, src, XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)) {
        printf("ERROR: Failed to invalidate BO buffer\n");
        exit(1);
    }
}

void* open_device()
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("Error opening /dev/mem\n");
        exit(1);
    }

    void *base_ptr = mmap(NULL, 0x100000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, CONTROL_ADDRESS);
    if (base_ptr == MAP_FAILED) {
        printf("Error mapping memory\n");
        exit(1);
    }

#if DEBUG
    printf("Mapped memory at %p\n", base_ptr);
#endif 

    close(fd);

    return base_ptr;
}

void *getBufferPhysicalAddress(xclDeviceHandle device, xclBufferHandle buffer)
{
    struct xclBOProperties props;
    xclGetBOProperties(device, buffer, &props);

    if (props.paddr >= 0x80000000) {
        printf("ERROR: Memory allocation did not success\n");
        exit(1);
    }

    return (void *)props.paddr;
}

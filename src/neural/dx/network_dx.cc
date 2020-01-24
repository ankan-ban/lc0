/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <algorithm>
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include "layers_dx.h"
#include "network_dx.h"
#include "neural/shared/policy_map.h"
#include "shader_wrapper.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#define DEFAULT_FP16 true
#define DEFAULT_ASYNC_COMPUTE false

namespace lczero {

using namespace dx_backend;

uint64_t DxContext::flushCL(ID3D12GraphicsCommandList5* cl) {
  if (!cl) cl = command_list_;
  cl->Close();
  command_queue_->ExecuteCommandLists(1, (ID3D12CommandList**)&cl);
  command_queue_->Signal(fence_, ++fence_val_);
  return fence_val_;
}

void DxContext::waitForGPU(uint64_t fence_val) {
  if (!fence_val) fence_val = fence_val_;
  // Wait for commands to finish on GPU.
  // (spinloop has lowest latency, we can try event based signal if CPU
  // overhead becomes a bottleneck).
  while (fence_->GetCompletedValue() != fence_val)
    ;
  upload_scratch_mem_.offset = 0;
}

void DxContext::resetCL(ID3D12GraphicsCommandList5* cl,
                        ID3D12CommandAllocator* ca, bool reset) {
  if (!cl) cl = command_list_;
  if (!ca) ca = command_allocator_;
  if (reset) {
    ca->Reset();
    cl->Reset(ca, NULL);
  }
  cl->SetDescriptorHeaps(1, &desc_heap_);
}

void DxContext::flushAndWait() {
  flushCL();
  waitForGPU();
  resetCL();
}

void DxContext::uavBarrier(ID3D12GraphicsCommandList5* command_list) {
  if (!command_list) command_list = command_list_;
  CD3DX12_RESOURCE_BARRIER uav_barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  command_list->ResourceBarrier(1, &uav_barrier);
}

void DxContext::dumpFp32(float* buf, int elements) {
  printf("\n");
  for (int i = 0; i < elements; i++) {
    printf("%8.4f ", buf[i]);
    if ((i % 8) == 7) printf("\n");
  }
  printf("\n");
}

void DxContext::copyTensor(DXAlloc dst, DXAlloc src, int bytes) {
  CD3DX12_RESOURCE_BARRIER barrier;

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      src.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_list_->ResourceBarrier(1, &barrier);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      dst.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_COPY_DEST);
  command_list_->ResourceBarrier(1, &barrier);

  command_list_->CopyBufferRegion(dst.pResource, dst.offset, src.pResource,
                                  src.offset, bytes);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      src.pResource, D3D12_RESOURCE_STATE_COPY_SOURCE,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_list_->ResourceBarrier(1, &barrier);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      dst.pResource, D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_list_->ResourceBarrier(1, &barrier);
}

void DxContext::dumpCpuTensor(void* data, int size, bool fp16,
                              bool allnewline) {
  printf("\n");
  float* fp32arr = (float*)data;
  uint16_t* arr = (uint16_t*)data;

  for (int i = 0; i < size; i++) {
    printf("%8.4f ", fp16 ? FP16toFP32(arr[i]) : fp32arr[i]);
    if (allnewline || ((i % 8) == 7)) printf("\n");
  }
  printf("\n");
}

void DxContext::dumpTensor(DXAlloc alloc, int size, bool fp16,
                           bool allnewline) {
  int bytes = size * (fp16 ? sizeof(dx_half) : sizeof(float));
  CD3DX12_RESOURCE_BARRIER barrier;
  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      alloc.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_COPY_SOURCE);
  command_list_->ResourceBarrier(1, &barrier);

  command_list_->CopyBufferRegion(readback_scratch_mem_.pResource,
                                  readback_scratch_mem_.offset, alloc.pResource,
                                  alloc.offset, bytes);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      alloc.pResource, D3D12_RESOURCE_STATE_COPY_SOURCE,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_list_->ResourceBarrier(1, &barrier);

  flushAndWait();
  void* cpuPtr;
  readback_scratch_mem_.pResource->Map(0, nullptr, &cpuPtr);
  dumpCpuTensor(cpuPtr, size, fp16, allnewline);
  readback_scratch_mem_.pResource->Unmap(0, nullptr);
}

DxContext::DxContext(const OptionsDict& options) {
  gpu_id_ = options.GetOrDefault<int>("gpu", 0);

  IDXGIFactory4* pFactory = nullptr;
  IDXGIAdapter* pAdapter = nullptr;
  ReportDxErrors(CreateDXGIFactory2(0, IID_PPV_ARGS(&pFactory)));
  ReportDxErrors(pFactory->EnumAdapters(gpu_id_, &pAdapter));
  pFactory->Release();

  if (!pAdapter) throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

  ReportDxErrors(D3D12CreateDevice(pAdapter, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&device_)));
  pAdapter->Release();

  D3D12_COMMAND_QUEUE_DESC commandqueueDesc;
  commandqueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  commandqueueDesc.NodeMask = 0;
  commandqueueDesc.Priority = 0;
  commandqueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  ReportDxErrors(device_->CreateCommandQueue(&commandqueueDesc,
                                             IID_PPV_ARGS(&command_queue_)));

  ReportDxErrors(device_->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator_)));

  ReportDxErrors(device_->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                            command_allocator_, NULL,
                                            IID_PPV_ARGS(&command_list_)));

  D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
  heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  heapDesc.NumDescriptors = kNumDescHeapSlots;
  ReportDxErrors(
      device_->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&desc_heap_)));

  command_list_->SetDescriptorHeaps(1, &desc_heap_);

  next_slot_in_desc_heap_ = 0;

  fence_val_ = 0ull;
  ReportDxErrors(device_->CreateFence(fence_val_, D3D12_FENCE_FLAG_NONE,
                                      IID_PPV_ARGS(&fence_)));

  shader_wrapper_.init(device_);

  // Allocate scratch space for uploads and read-back.
  CreateAlloc(kUploadDownloadScratchSize, D3D12_HEAP_TYPE_UPLOAD,
              upload_scratch_mem_, false);
  CreateAlloc(kUploadDownloadScratchSize, D3D12_HEAP_TYPE_READBACK,
              readback_scratch_mem_, false);
}

DxContext::~DxContext() {
  // Make sure nothing is in flight
  flushAndWait();

  upload_scratch_mem_.pResource->Release();
  readback_scratch_mem_.pResource->Release();

  shader_wrapper_.destroy();
  command_list_->Release();
  command_allocator_->Release();
  command_queue_->Release();
  fence_->Release();
  desc_heap_->Release();
  device_->Release();
}

void DxContext::CreateAlloc(size_t size, D3D12_HEAP_TYPE type, DXAlloc& alloc,
                            bool fp16) {
  // some alignment
  int factor = DivUp((int)size, 4);
  size = factor * 4;

  D3D12_HEAP_PROPERTIES heapDesc = {};
  heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapDesc.CreationNodeMask = 1;
  heapDesc.VisibleNodeMask = 1;

  if (type == D3D12_HEAP_TYPE_CUSTOM) {
    // Use custom heap type to allow GPU writing to system memory directly
    heapDesc.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    heapDesc.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
  }

  heapDesc.Type = type;

  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.Height = 1;
  if (type == D3D12_HEAP_TYPE_DEFAULT || type == D3D12_HEAP_TYPE_CUSTOM)
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  D3D12_RESOURCE_STATES resourceState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  if (type == D3D12_HEAP_TYPE_UPLOAD)
    resourceState = D3D12_RESOURCE_STATE_GENERIC_READ;
  else if (type == D3D12_HEAP_TYPE_READBACK)
    resourceState = D3D12_RESOURCE_STATE_COPY_DEST;

  bufferDesc.Width = size;
  ReportDxErrors(device_->CreateCommittedResource(
      &heapDesc, D3D12_HEAP_FLAG_NONE, &bufferDesc, resourceState, nullptr,
      IID_PPV_ARGS(&alloc.pResource)));

  alloc.offset = 0;
  alloc.gpuVA = alloc.pResource->GetGPUVirtualAddress();

  // Create desc heap entries for UAV resources.
  if (resourceState == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
    int handleIncrementSize = device_->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    size_t element_size = fp16 ? sizeof(dx_half) : sizeof(float);

    // Scalar UAV.
    {
      int slot = next_slot_in_desc_heap_++;

      CD3DX12_CPU_DESCRIPTOR_HANDLE cpuDescHandle(
          desc_heap_->GetCPUDescriptorHandleForHeapStart(), slot,
          handleIncrementSize);

      CD3DX12_GPU_DESCRIPTOR_HANDLE gpuDescHandle(
          desc_heap_->GetGPUDescriptorHandleForHeapStart(), slot,
          handleIncrementSize);

      D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
      uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      uavDesc.Format = fp16 ? DXGI_FORMAT_R16_FLOAT : DXGI_FORMAT_R32_FLOAT;
      uavDesc.Buffer.FirstElement = 0;
      uavDesc.Buffer.NumElements = (UINT)(size / element_size);

      device_->CreateUnorderedAccessView(alloc.pResource, nullptr, &uavDesc,
                                         cpuDescHandle);

      alloc.descHandleScalar = gpuDescHandle;
    }

    // 4-component vector UAV.
    {
      int slot = next_slot_in_desc_heap_++;

      CD3DX12_CPU_DESCRIPTOR_HANDLE cpuDescHandle(
          desc_heap_->GetCPUDescriptorHandleForHeapStart(), slot,
          handleIncrementSize);

      CD3DX12_GPU_DESCRIPTOR_HANDLE gpuDescHandle(
          desc_heap_->GetGPUDescriptorHandleForHeapStart(), slot,
          handleIncrementSize);

      D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
      uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
      uavDesc.Format = fp16 ? DXGI_FORMAT_R16G16B16A16_FLOAT
                            : DXGI_FORMAT_R32G32B32A32_FLOAT;
      uavDesc.Buffer.FirstElement = 0;
      uavDesc.Buffer.NumElements = (UINT)(size / (4 * element_size));

      device_->CreateUnorderedAccessView(alloc.pResource, nullptr, &uavDesc,
                                         cpuDescHandle);

      alloc.descHandleVector = gpuDescHandle;
    }
  }
}

void DxContext::scheduleUpload(DXAlloc alloc, const void* data, size_t size) {
  // Make sure enough space is available in the upload scratch buffer
  assert(size <= kUploadDownloadScratchSize);
  if (upload_scratch_mem_.offset + size > kUploadDownloadScratchSize)
    flushAndWait();

  uint8_t* temp;
  upload_scratch_mem_.pResource->Map(0, nullptr, (void**)&temp);

  dx_half* cpuPtr = (dx_half*)(temp + upload_scratch_mem_.offset);
  memcpy(cpuPtr, data, size);

  CD3DX12_RESOURCE_BARRIER barrier;

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      alloc.pResource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      D3D12_RESOURCE_STATE_COPY_DEST);
  command_list_->ResourceBarrier(1, &barrier);

  command_list_->CopyBufferRegion(alloc.pResource, alloc.offset,
                                  upload_scratch_mem_.pResource,
                                  upload_scratch_mem_.offset, size);

  barrier = CD3DX12_RESOURCE_BARRIER::Transition(
      alloc.pResource, D3D12_RESOURCE_STATE_COPY_DEST,
      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  command_list_->ResourceBarrier(1, &barrier);

  upload_scratch_mem_.pResource->Unmap(0, nullptr);

  // reset at flush and wait
  upload_scratch_mem_.offset += (uint32_t)size;
}

DxNetwork::DxNetwork(const WeightsFile& file, const OptionsDict& options)
    : dx_context_(options),
      capabilities_{file.format().network_format().input()} {
  LegacyWeights weights(file.weights());

  has_conv_policy_ = file.format().network_format().policy() ==
                     pblczero::NetworkFormat::POLICY_CONVOLUTION;
  max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

  // Default is fp16, to use fp32: --backend-opts=fp16=false.
  fp16_ = options.GetOrDefault<bool>("fp16", DEFAULT_FP16);

  // Default is to attempt using Winograd algorithm for Convolutions using GEMM
  // Metacommand first, if not available - attempt using Convolution Metacommand
  // directly (whatever algorithm HW vendor is providing), and if neither is
  // available use winograd algorithm with our own GEMM compute shader.
  // The below backend options can be used to override this for testing.
  bool enable_gemm_metacommand =
      options.GetOrDefault<bool>("enable-gemm-metacommand", true);
  bool enable_conv_metacommand =
      options.GetOrDefault<bool>("enable-conv-metacommand", true);

  // When async compute is enabled, we create a new command queue for each
  // "InputsOutputs" instance. The first one is created as a regular (Direct)
  // queue and subsequent instances get created acync compute queues. This will
  // result in greater memory usage as memory for intermediate tensor storage
  // also needs to be allocated per command queue, but it can improve
  // performance as GPU can be parallely evaluating two calls at the same time
  // better utilizaing the resources.
  enable_async_compute_ =
      options.GetOrDefault<bool>("enable-async-compute", DEFAULT_ASYNC_COMPUTE);
  num_queues_created_ = 0;

  const int kNumFilters = (int)weights.input.biases.size();

  num_blocks_ = (int)weights.residual.size();
  has_se_ = weights.residual[0].has_se;
  int pol_channels = (int)weights.policy.biases.size();

  // Build the network, and copy the weights to GPU memory.

  // Unique GEMMs for winograd required by the network.
  if (enable_gemm_metacommand) {
    input_conv_gemm_metacommand_ = std::make_unique<GemmMetaCommand>(
        &dx_context_, 0, kNumFilters, kInputPlanes, 36, fp16_, false, false);

    residual_block_gemm_metacommand_ = std::make_unique<GemmMetaCommand>(
        &dx_context_, 0, kNumFilters, kNumFilters, 36, fp16_, false, false);

    if (has_conv_policy_) {
      policy_conv_gemm_metacommand_ = std::make_unique<GemmMetaCommand>(
          &dx_context_, 0, pol_channels, kNumFilters, 36, fp16_, false, false);
    }
  }

  // Unique Conv metacommands required by the network.
  if (enable_conv_metacommand) {
    // Create only if we were not able to create GEMM metacommands for some
    // reason 3x3, 112 channels -> kNumFilters channels, relu, bias.
    if (!input_conv_gemm_metacommand_ ||
        !input_conv_gemm_metacommand_->IsAvailable())
      input_conv_metacommand_ = std::make_unique<ConvMetaCommand>(
          &dx_context_, kInputPlanes, kNumFilters, 8, 8, 3, true, true, fp16_);

    if (!residual_block_gemm_metacommand_ ||
        !residual_block_gemm_metacommand_->IsAvailable()) {
      // 3x3, kNumFilters channels -> kNumFilters channels, relu, bias.
      resi_block_conv_1_metacommand_ = std::make_unique<ConvMetaCommand>(
          &dx_context_, kNumFilters, kNumFilters, 8, 8, 3, true, true, fp16_);

      // 3x3, kNumFilters channels -> kNumFilters channels, no relu
      // relu needs to be done after SE and/or skip connection add.
      resi_block_conv_2_metacommand_ = std::make_unique<ConvMetaCommand>(
          &dx_context_, kNumFilters, kNumFilters, 8, 8, 3, false, true, fp16_);
    }

    if (has_conv_policy_ && (!policy_conv_gemm_metacommand_ ||
                             !policy_conv_gemm_metacommand_->IsAvailable()))
      policy_conv_metacommand_ = std::make_unique<ConvMetaCommand>(
          &dx_context_, kNumFilters, pol_channels, 8, 8, 3, false, true, fp16_);
  }

  // input
  {
    auto inputConv = std::make_unique<ConvLayer>(
        fp16_, input_conv_gemm_metacommand_.get(),
        input_conv_metacommand_.get(), &dx_context_, nullptr, kNumFilters, 8, 8,
        3, kInputPlanes, true, true);

    inputConv->LoadWeights(&weights.input.weights[0], &weights.input.biases[0],
                           &dx_context_);

    network_.emplace_back(std::move(inputConv));
  }

  // residual block
  for (size_t block = 0; block < weights.residual.size(); block++) {
    auto conv1 = std::make_unique<ConvLayer>(
        fp16_, residual_block_gemm_metacommand_.get(),
        resi_block_conv_1_metacommand_.get(), &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true);

    conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                       &weights.residual[block].conv1.biases[0], &dx_context_);

    network_.emplace_back(std::move(conv1));

    int se_k = 0;
    if (has_se_) se_k = (int)weights.residual[block].se.b1.size();

    auto conv2 = std::make_unique<ConvLayer>(
        fp16_, residual_block_gemm_metacommand_.get(),
        resi_block_conv_2_metacommand_.get(), &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true, true, has_se_, se_k);

    conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                       &weights.residual[block].conv2.biases[0], &dx_context_);

    if (has_se_) {
      conv2->LoadSEWeights(
          &weights.residual[block].se.w1[0], &weights.residual[block].se.b1[0],
          &weights.residual[block].se.w2[0], &weights.residual[block].se.b2[0]);
    }
    network_.emplace_back(std::move(conv2));
  }

  BaseLayer* resi_last = getLastLayer();

  // policy head
  if (has_conv_policy_) {
    // conv1 is same as residual block convolution.
    auto conv1 = std::make_unique<ConvLayer>(
        fp16_, residual_block_gemm_metacommand_.get(),
        resi_block_conv_1_metacommand_.get(), &dx_context_, getLastLayer(),
        kNumFilters, 8, 8, 3, kNumFilters, true, true);
    conv1->LoadWeights(&weights.policy1.weights[0], &weights.policy1.biases[0],
                       &dx_context_);
    network_.emplace_back(std::move(conv1));

    // conv2 has different no. of output filters (pol_channels). No relu.
    auto conv2 = std::make_unique<ConvLayer>(
        fp16_, policy_conv_gemm_metacommand_.get(),
        policy_conv_metacommand_.get(), &dx_context_, getLastLayer(),
        pol_channels, 8, 8, 3, kNumFilters, true, false);

    conv2->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                       &dx_context_);
    network_.emplace_back(std::move(conv2));

    // Policy map layer
    auto policyMap =
        std::make_unique<PolicyMapLayer>(fp16_, &dx_context_, getLastLayer(),
                                         kNumOutputPolicy, 1, 1, 73 * 8 * 8);
    policyMap->LoadWeights(kConvPolicyMap);
    network_.emplace_back(std::move(policyMap));

  } else {
    // 1x1 convolution, pol_channels output filters
    auto convPol = std::make_unique<ConvLayer>(
        fp16_, nullptr, nullptr, &dx_context_, getLastLayer(), pol_channels, 8,
        8, 1, kNumFilters, true, true);
    convPol->LoadWeights(&weights.policy.weights[0], &weights.policy.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convPol));

    // FC with bias, no activation
    // pad up kNumOutputPolicy to be a multiple of 8
    assert(weights.ip_pol_b.size() == kNumOutputPolicy);
    auto FCPol = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                           kNumOutputPolicyPadded8, 1, 1, true,
                                           false, false);
    // Copy weights to temp space which is padded in size.
    std::vector<float> tempBias(kNumOutputPolicyPadded8);
    std::vector<float> tempWeight(kNumOutputPolicyPadded8 *
                                  weights.ip_pol_w.size() / kNumOutputPolicy);
    memcpy(tempBias.data(), weights.ip_pol_b.data(),
           weights.ip_pol_b.size() * sizeof(float));
    memcpy(tempWeight.data(), weights.ip_pol_w.data(),
           weights.ip_pol_w.size() * sizeof(float));

    FCPol->LoadWeights(tempWeight.data(), tempBias.data(), &dx_context_);
    network_.emplace_back(std::move(FCPol));
  }

  // value head
  {
    int val_channels = (int)weights.value.biases.size();

    // 1x1 convolution, val_channels output filters
    auto convVal = std::make_unique<ConvLayer>(
        fp16_, nullptr, nullptr, &dx_context_, getLastLayer(), val_channels, 8,
        8, 1, kNumFilters, true, true);
    convVal->LoadWeights(&weights.value.weights[0], &weights.value.biases[0],
                         &dx_context_);
    network_.emplace_back(std::move(convVal));

    // Bias and relu activation.
    auto FCVal1 = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                            (int)weights.ip1_val_b.size(), 1, 1,
                                            true, true, false);
    FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                        &dx_context_);
    network_.emplace_back(std::move(FCVal1));

    has_wdl_ = file.format().network_format().value() ==
               pblczero::NetworkFormat::VALUE_WDL;

    // Fully connected layer with Bias.
    // tanh activation for non wdl nets, no activation for wdl.
    auto fc2_tanh = !has_wdl_;
    auto FCVal2 = std::make_unique<FCLayer>(fp16_, &dx_context_, getLastLayer(),
                                            kNumOutputValuePadded8, 1, 1, true,
                                            false, fc2_tanh);
    // Pad up the weights
    std::vector<float> tempBias(kNumOutputValuePadded8);
    std::vector<float> tempWeight(kNumOutputValuePadded8 *
                                  weights.ip2_val_w.size() /
                                  weights.ip2_val_b.size());
    memcpy(tempBias.data(), weights.ip2_val_b.data(),
           weights.ip2_val_b.size() * sizeof(float));
    memcpy(tempWeight.data(), weights.ip2_val_w.data(),
           weights.ip2_val_w.size() * sizeof(float));
    FCVal2->LoadWeights(tempWeight.data(), tempBias.data(), &dx_context_);
    network_.emplace_back(std::move(FCVal2));
  }

  dx_context_.flushAndWait();

  // Allocate GPU memory for running the network
  // 4 buffers of max size are enough:
  //   * one to hold input,
  //   * second to hold output
  //   * third to hold skip connection's input
  //   * and fourth to act as scratch space needed by some layers.
  size_t max_size = resi_last->GetOutputSize(max_batch_size_);

  // Winograd transformed inputs/outputs need more space.
  // Every 4x4 block of input/output is transfored to 6x6 block.
  max_size *= (size_t)ceil(36.0 / 16.0);
  tensor_mem_size_ = max_size;

  if (!enable_async_compute_) {
    for (auto& mem : tensor_mem_)
      dx_context_.CreateAlloc(max_size, D3D12_HEAP_TYPE_DEFAULT, mem, fp16_);
  }
}

void DxNetwork::Eval(InputsOutputsDx* io, int batchSize) {
  if (batchSize > kMaxSupportedBatchSize)
    throw Exception("Unsupported batch size: " + std::to_string(batchSize));

  PerThreadData* dt = per_thread_data_[std::this_thread::get_id()].get();

  ID3D12GraphicsCommandList5* cl = dt->command_list_;
  dx_context_.resetCL(cl, dt->command_allocator_, dt->needs_reset_);

  // Expand packed board representation into full planes.
  dx_context_.getShaderWrapper()->expandPlanes(
      cl, dt->tensor_mem_[0], io->input_masks_mem_gpu_, io->input_val_mem_gpu_,
      batchSize, fp16_);
  dx_context_.uavBarrier(cl);

  // Debug dumping example.
  // printf("\nAfter expand planes");
  // dx_context_.dumpTensor(dt->tensor_mem_[0], 1024, fp16_);

  int l = 0;

  //-----------------------------------///---------------------------------------
  // Input Conv
  network_[l++]->Eval(batchSize, dt->tensor_mem_[2], dt->tensor_mem_[0],
                      DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
  dx_context_.uavBarrier(cl);

  //-----------------------------------///---------------------------------------

  // Residual tower.
  for (int block = 0; block < num_blocks_; block++) {
    // conv1
    network_[l++]->Eval(batchSize, dt->tensor_mem_[0], dt->tensor_mem_[2],
                        DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
    dx_context_.uavBarrier(cl);

    // conv2
    network_[l++]->Eval(batchSize, dt->tensor_mem_[2], dt->tensor_mem_[0],
                        dt->tensor_mem_[2], dt->tensor_mem_[1],
                        dt->tensor_mem_[3], cl);
    dx_context_.uavBarrier(cl);
  }

  //-----------------------------------///---------------------------------------

  // Policy head.
  if (has_conv_policy_) {
    // Policy conv1.
    network_[l++]->Eval(batchSize, dt->tensor_mem_[0], dt->tensor_mem_[2],
                        DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
    dx_context_.uavBarrier(cl);

    // Policy conv2
    network_[l++]->Eval(batchSize, dt->tensor_mem_[1], dt->tensor_mem_[0],
                        DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);

    dx_context_.uavBarrier(cl);

    // Policy Map layer  (writes directly to system memory).
    network_[l++]->Eval(batchSize, io->op_policy_mem_gpu_, dt->tensor_mem_[1],
                        DXAlloc(), DXAlloc(), DXAlloc(), cl);
  } else {
    // Policy conv.
    network_[l++]->Eval(batchSize, dt->tensor_mem_[0], dt->tensor_mem_[2],
                        DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
    dx_context_.uavBarrier(cl);

    // Policy FC (writes directly to system memory).
    network_[l++]->Eval(batchSize, io->op_policy_mem_gpu_, dt->tensor_mem_[0],
                        DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
  }

  //-----------------------------------///---------------------------------------

  // Value head.

  // Value conv.
  network_[l++]->Eval(batchSize, dt->tensor_mem_[0], dt->tensor_mem_[2],
                      DXAlloc(), dt->tensor_mem_[1], dt->tensor_mem_[3], cl);
  dx_context_.uavBarrier(cl);

  // value FC1.
  network_[l++]->Eval(batchSize, dt->tensor_mem_[1], dt->tensor_mem_[0],
                      DXAlloc(), dt->tensor_mem_[2], dt->tensor_mem_[3], cl);
  dx_context_.uavBarrier(cl);

  // value FC2.
  network_[l++]->Eval(batchSize, io->op_value_mem_gpu_, dt->tensor_mem_[1],
                      DXAlloc(), dt->tensor_mem_[2], dt->tensor_mem_[3], cl);

  //-----------------------------------///---------------------------------------

  // TODO: Get rid of this lock once we move the Command Queue also to
  // InputsOutputs structure This isn't a bottleneck anyway (for CPU side perf).
  // The hope is that we will get some GPU side parallelism with multiple async
  // compute queues.
  if (enable_async_compute_ && dt->queue_index_ > 0) {
    cl->Close();
    dt->command_queue_->ExecuteCommandLists(1, (ID3D12CommandList**)&cl);
    dt->command_queue_->Signal(dt->fence_, ++dt->fence_val_);
    while (dt->fence_->GetCompletedValue() != dt->fence_val_)
      ;
  } else {
    lock_.lock();
    uint64_t fence = dx_context_.flushCL(cl);
    lock_.unlock();
    dx_context_.waitForGPU(fence);
  }
  dt->needs_reset_ = true;

  // Do some simple post-processing operations on CPU:
  // - un-padding of policy and value heads.
  // - value head softmax (for wdl enabled nets)
  // We do them outside the lock to get some more parallelism.
  int val_vector_size = has_wdl_ ? 3 : 1;
  if (fp16_) {
    // Policy:
    // Un-pad policy output, and convert to fp32.
    if (!has_conv_policy_) {
      dx_half* padded_pol_fp16 = (dx_half*)io->op_policy_mem_;
      for (int n = 0; n < batchSize; n++)
        for (int i = 0; i < kNumOutputPolicy; i++)
          io->op_policy_mem_final_[n * kNumOutputPolicy + i] =
              FP16toFP32(padded_pol_fp16[n * kNumOutputPolicyPadded8 + i]);
    }
    // Value:
    // Un-pad value output, converting it to fp32.
    dx_half* padded_val_fp16 = (dx_half*)io->op_value_mem_;
    for (int n = 0; n < batchSize; n++)
      for (int i = 0; i < val_vector_size; i++)
        io->op_value_mem_final_[n * val_vector_size + i] =
            FP16toFP32(padded_val_fp16[n * kNumOutputValuePadded8 + i]);

  } else {
    // Policy:
    // Un-pad policy output.
    if (!has_conv_policy_) {
      for (int i = 0; i < batchSize; i++)
        memcpy(io->op_policy_mem_final_ + kNumOutputPolicy * i,
               io->op_policy_mem_ + kNumOutputPolicyPadded8 * i,
               kNumOutputPolicy * sizeof(float));
    }

    // Value:
    // Un-pad value output.
    for (int i = 0; i < batchSize; i++)
      memcpy(io->op_value_mem_final_ + val_vector_size * i,
             io->op_value_mem_ + kNumOutputValuePadded8 * i,
             val_vector_size * sizeof(float));
  }

  // Softmax on value head for wdl enabled networks.
  if (has_wdl_) {
    for (int i = 0; i < batchSize; i++) {
      float w_val = io->op_value_mem_final_[i * 3 + 0];
      float d_val = io->op_value_mem_final_[i * 3 + 1];
      float l_val = io->op_value_mem_final_[i * 3 + 2];

      w_val = exp(w_val);
      d_val = exp(d_val);
      l_val = exp(l_val);
      float S = w_val + d_val + l_val;
      w_val /= S;
      d_val /= S;
      l_val /= S;

      io->op_value_mem_final_[i * 3 + 0] = w_val;
      io->op_value_mem_final_[i * 3 + 1] = d_val;
      io->op_value_mem_final_[i * 3 + 2] = l_val;
    }
  }
}

DxNetwork::~DxNetwork() {
  dx_context_.flushAndWait();
  // Free memory and destroy all dx objects.
  if (!enable_async_compute_)
    for (auto mem : tensor_mem_) mem.pResource->Release();
}

std::unique_ptr<NetworkComputation> DxNetwork::NewComputation() {
  return std::make_unique<DxNetworkComputation>(this, has_wdl_);
}

std::unique_ptr<InputsOutputsDx> DxNetwork::GetInputsOutputs() {
  std::lock_guard<std::mutex> lock(inputs_outputs_lock_);

  std::thread::id tid = std::this_thread::get_id();
  PerThreadData* data = per_thread_data_[tid].get();
  if (!data) {
    per_thread_data_[tid] = std::make_unique<PerThreadData>(this);
    data = per_thread_data_[tid].get();
  }

  if (free_inputs_outputs_.empty()) {
    return std::make_unique<InputsOutputsDx>(this);
  } else {
    std::unique_ptr<InputsOutputsDx> resource =
        std::move(free_inputs_outputs_.front());
    free_inputs_outputs_.pop_front();
    return resource;
  }
}

void DxNetwork::ReleaseInputsOutputs(
    std::unique_ptr<InputsOutputsDx> resource) {
  std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
  free_inputs_outputs_.push_back(std::move(resource));
}

DxNetworkComputation::DxNetworkComputation(DxNetwork* network, bool wdl)
    : network_(network), wdl_(wdl) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

DxNetworkComputation::~DxNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

void DxNetworkComputation::AddInput(InputPlanes&& input) {
  auto iter_mask =
      &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
  auto iter_val = &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

  int i = 0;
  for (const auto& plane : input) {
    iter_mask[i] = plane.mask;
    iter_val[i] = plane.value;
    i++;
  }

  batch_size_++;
}

void DxNetworkComputation::ComputeBlocking() {
  network_->Eval(inputs_outputs_.get(), GetBatchSize());
}

PerThreadData::PerThreadData(DxNetwork* network) {
  uses_async_compute_ = network->enable_async_compute_;
  needs_reset_ = true;

  printf("\n Creating per thread data \n");
  DxContext* dx_context = &(network->dx_context_);
  ID3D12Device* device = dx_context->getDevice();

  queue_index_ = network->num_queues_created_++;
  queue_index_++;   // Ankan - hack! put all work in compute queues.
  D3D12_COMMAND_LIST_TYPE cl_type = (uses_async_compute_ && (queue_index_ > 0))
                                        ? D3D12_COMMAND_LIST_TYPE_COMPUTE
                                        : D3D12_COMMAND_LIST_TYPE_DIRECT;

  ReportDxErrors(device->CreateCommandAllocator(
      cl_type, IID_PPV_ARGS(&command_allocator_)));

  ReportDxErrors(device->CreateCommandList(1, cl_type, command_allocator_, NULL,
                                           IID_PPV_ARGS(&command_list_)));

  if (uses_async_compute_) {
    if (queue_index_ > 0) {
      D3D12_COMMAND_QUEUE_DESC commandqueueDesc;
      commandqueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
      commandqueueDesc.NodeMask = 0;
      commandqueueDesc.Priority = 0;
      commandqueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
      ReportDxErrors(device->CreateCommandQueue(&commandqueueDesc,
                                                IID_PPV_ARGS(&command_queue_)));

      fence_val_ = 0;
      ReportDxErrors(device->CreateFence(fence_val_, D3D12_FENCE_FLAG_NONE,
                                         IID_PPV_ARGS(&fence_)));
    } else {
      command_queue_ = dx_context->getCommandQueue();
      fence_ = dx_context->getFence();
      fence_val_ = dx_context->getFenceVal();
    }

    for (auto& mem : tensor_mem_)
      dx_context->CreateAlloc(network->tensor_mem_size_,
                              D3D12_HEAP_TYPE_DEFAULT, mem, network->fp16_);
  } else {
    int i = 0;
    for (auto& mem : tensor_mem_) mem = network->tensor_mem_[i++];
  }
}
PerThreadData::~PerThreadData() {
  command_allocator_->Release();
  command_list_->Release();

  if (uses_async_compute_) {
    if (queue_index_ > 0) {
      command_queue_->Release();
      fence_->Release();
    }
    for (auto& mem : tensor_mem_) {
      mem.pResource->Release();
    }
  }
}

InputsOutputsDx::InputsOutputsDx(DxNetwork* network)
    : uses_policy_map_(network->has_conv_policy_) {
  DxContext* dx_context = &(network->dx_context_);
  int max_batch = network->max_batch_size_;
  bool fp16 = network->fp16_;
  bool wdl = network->has_wdl_;

  printf("\nCreating a new InputsOutputsDx\n");

  // CPU accesses on Default heap doesn't work.
  // GPU accesses on Upload heap works.
  dx_context->CreateAlloc(max_batch * kInputPlanes * sizeof(uint64_t),
                          D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                          input_masks_mem_gpu_, fp16);

  dx_context->CreateAlloc(max_batch * kInputPlanes * sizeof(float),
                          D3D12_HEAP_TYPE_UPLOAD /*D3D12_HEAP_TYPE_DEFAULT*/,
                          input_val_mem_gpu_, fp16);

  // CUSTOM heap created to have GPU directly write to system memory
  dx_context->CreateAlloc(max_batch * kNumOutputPolicyPadded8 * sizeof(float),
                          D3D12_HEAP_TYPE_CUSTOM, op_policy_mem_gpu_, fp16);

  dx_context->CreateAlloc(max_batch * kNumOutputValuePadded8 * sizeof(float),
                          D3D12_HEAP_TYPE_CUSTOM, op_value_mem_gpu_, fp16);

  ReportDxErrors(input_masks_mem_gpu_.pResource->Map(
      0, nullptr, (void**)&input_masks_mem_));

  ReportDxErrors(
      input_val_mem_gpu_.pResource->Map(0, nullptr, (void**)&input_val_mem_));

  ReportDxErrors(
      op_policy_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_policy_mem_));

  ReportDxErrors(
      op_value_mem_gpu_.pResource->Map(0, nullptr, (void**)&op_value_mem_));

  // When policy map is enabled, GPU writes directly to the final policy output.
  if (uses_policy_map_)
    op_policy_mem_final_ = op_policy_mem_;
  else
    op_policy_mem_final_ = new float[max_batch * kNumOutputPolicy];
  op_value_mem_final_ = new float[max_batch * (wdl ? 3 : 1)];
}

InputsOutputsDx::~InputsOutputsDx() {
  printf("\nInputsOutputsDx destroyed\n");
  input_masks_mem_gpu_.pResource->Unmap(0, nullptr);
  input_val_mem_gpu_.pResource->Unmap(0, nullptr);
  op_policy_mem_gpu_.pResource->Unmap(0, nullptr);
  op_value_mem_gpu_.pResource->Unmap(0, nullptr);

  input_masks_mem_gpu_.pResource->Release();
  input_val_mem_gpu_.pResource->Release();
  op_policy_mem_gpu_.pResource->Release();
  op_value_mem_gpu_.pResource->Release();

  if (!uses_policy_map_) delete[] op_policy_mem_final_;
  delete[] op_value_mem_final_;
}

std::unique_ptr<Network> MakeDxNetwork(const WeightsFile& weights,
                                       const OptionsDict& options) {
  return std::make_unique<DxNetwork>(weights, options);
}

REGISTER_NETWORK("dx", MakeDxNetwork, 120)

}  // namespace lczero

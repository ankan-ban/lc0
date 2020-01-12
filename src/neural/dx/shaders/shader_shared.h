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

#define kExpandPlanesElementsPerBlock 256
#define kExpandPlanesFp32BlockSize kExpandPlanesElementsPerBlock
#define kExpandPlanesFp16BlockSize (kExpandPlanesElementsPerBlock / 2)

// for both input transform and output transform shaders
#define kWinogradTransformShaderBlockSize 64

#define kConv1x1BlockSize 64

#define kAddVectorsBlockSize 512

#define kPolicyMapBlockSize 256


// Constants for GEMM shader.
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

#define ELEMENTS_PER_THREAD_X 8
#define ELEMENTS_PER_THREAD_Y 8

#define ELEMENTS_PER_BLOCK_X (ELEMENTS_PER_THREAD_X * BLOCK_WIDTH)
#define ELEMENTS_PER_BLOCK_Y (ELEMENTS_PER_THREAD_Y * BLOCK_HEIGHT)

#define SHARED_MEM_K_CHUNK 16
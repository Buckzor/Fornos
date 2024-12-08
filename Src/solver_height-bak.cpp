/*
Copyright 2018 Oscar Sebio Cajaraville

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "solver_height.h"
#include "bvh.h"
#include "compute.h"
#include "computeshaders.h"
#include "image.h"
#include "logging.h"
#include "math.h"
#include "mesh.h"
#include "meshmapping.h"
#include <vector>
#include <cmath>
#include <cstdlib> // For rand()
#include <cassert>
#include <iostream>

static const size_t k_groupSize = 64;
static const size_t k_workPerFrame = 1024 * 128;

namespace
{
	std::vector<Vector3> computeSamples(size_t sampleCount, size_t permutationCount)
	{
		const size_t count = sampleCount * permutationCount;
		std::vector<Vector3> sampleDirs(count);
		computeSamplesImportanceCosDir(sampleCount, permutationCount, &sampleDirs[0]);
		return sampleDirs;
	}
}

float randomFloat()
{
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float degreesToRadians(float degrees)
{
	return degrees * (PI / 180);
}

std::vector<Vector4> computeSampleDirections(uint32_t sampleCount, float coneAngleDegrees)
{
	std::vector<Vector4> sampleDirs(sampleCount);

	float coneAngleRadians = degreesToRadians(coneAngleDegrees);
	float cosConeAngle = cosf(coneAngleRadians);

	for (uint32_t i = 0; i < sampleCount; ++i)
	{
		// Generate a random direction within the cone
		// For simplicity, we'll use uniformly distributed random directions within the cone
		// You might want to use a more sophisticated sampling method for better distribution

		float u1 = randomFloat(); // Function to generate random float between 0 and 1
		float u2 = randomFloat();

		float theta = acosf(1.0f - u1 + u1 * cosConeAngle);
		float phi = 2.0f * PI * u2;

		float sinTheta = sinf(theta);

		Vector4 dir;
		dir.x = sinTheta * cosf(phi);
		dir.y = sinTheta * sinf(phi);
		dir.z = cosf(theta);
		dir.w = 0.0f;

		// Store the direction as a Vector4 (the w component can be unused or set to 0)
		sampleDirs[i] = dir;
	}

	return sampleDirs;
}

void HeightSolver::init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<MeshMapping> meshMapping)
{
    _heightProgram = LoadComputeShader_Height();
    _uvMap = map;
    _meshMapping = meshMapping;
    _workCount = ((map->positions.size() + k_groupSize - 1) / k_groupSize) * k_groupSize;
    _resultsCB = std::unique_ptr<ComputeBuffer<float>>(
        new ComputeBuffer<float>(_workCount, GL_STATIC_READ));
	_debugCB = std::unique_ptr<ComputeBuffer<uint32_t>>(
		new ComputeBuffer<uint32_t>(_workCount, GL_STATIC_READ));
    _workOffset = 0;

	// Generate sample directions
	auto sampleDirs = computeSampleDirections(_params.sampleCount, _params.coneAngle);
	_samplesCB = std::unique_ptr<ComputeBuffer<Vector4>>(
		new ComputeBuffer<Vector4>(&sampleDirs[0], sampleDirs.size(), GL_STATIC_DRAW));
}

bool HeightSolver::runStep()
{
	assert(_workOffset < _workCount);
	const size_t workLeft = _workCount - _workOffset;
	const size_t work = workLeft < k_workPerFrame ? workLeft : k_workPerFrame;
	assert(work % k_groupSize == 0);

	if (_workOffset == 0) _timing.begin();

	glUseProgram(_heightProgram);

	// Set uniforms
	glUniform1ui(1, (GLuint)_workOffset);           // workOffset
	glUniform1ui(2, _params.sampleCount);           // sampleCount
	glUniform1f(3, _params.maxDistance);            // maxDistance
	glUniform1f(4, _params.coneAngle);              // coneAngleDegrees
	glUniform1ui(5, (GLuint)_meshMapping->meshBVH()->size()); // bvhCount

	// Bind buffers
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _meshMapping->coords()->bo());      // coordsBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _resultsCB->bo());                  // resultBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _samplesCB->bo());                  // samplesBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _meshMapping->meshPositions()->bo()); // meshPBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _meshMapping->meshBVH()->bo());     // bvhBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _meshMapping->meshNormals()->bo()); // meshNBuffer
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _debugCB->bo()); // debugBuffer

	// Dispatch compute shader
	glDispatchCompute((GLuint)(work / k_groupSize), 1, 1);

	_workOffset += work;

	if (_workOffset >= _workCount)
	{
		_timing.end();
		logDebug("Height",
			"Height map took " + std::to_string(_timing.elapsedSeconds()) +
			" seconds for " + std::to_string(_uvMap->width) + "x" + std::to_string(_uvMap->height));
	}

	return _workOffset >= _workCount;
}

float* HeightSolver::getResults()
{
	assert(_workOffset == _workCount);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	// Read back the results buffer
	float* results = new float[_workCount];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, _resultsCB->bo());
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * _workCount, results);

	// Read back the debug buffer
	uint32_t* hitCounts = new uint32_t[_workCount];
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, _debugCB->bo());
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t) * _workCount, hitCounts);

	// Log the first few hitCounts using logDebug
	std::string hitCountsLog = "Hit counts for the first 10 pixels:";
	for (size_t i = 0; i < 100 && i < _workCount; ++i)
	{
		hitCountsLog += "\nPixel " + std::to_string(i) + ": " + std::to_string(hitCounts[i]);
	}
	logDebug("Height", hitCountsLog);

	// Optionally, you can log summary statistics
	size_t totalHits = 0;
	for (size_t i = 0; i < _workCount; ++i)
	{
		totalHits += hitCounts[i];
	}
	float averageHits = static_cast<float>(totalHits) / _workCount;
	logDebug("Height", "Average hit count per pixel: " + std::to_string(averageHits));

	delete[] hitCounts;

	return results;
}


HeightTask::HeightTask(std::unique_ptr<HeightSolver> solver, const char *outputPath, int dilation)
	: _solver(std::move(solver))
	, _outputPath(outputPath)
	, _dilation(dilation)
{
}

HeightTask::~HeightTask()
{
}

bool HeightTask::runStep()
{
	assert(_solver);
	return _solver->runStep();
}

void HeightTask::finish()
{
	assert(_solver);
	float *results = _solver->getResults();
	auto map = _solver->uvMap();
	Vector2 minmax;
	exportFloatImage(
		results,
		_solver->uvMap().get(),
		_outputPath.c_str(),
		Vector2(0, _solver->parameters().maxDistance),
		_solver->parameters().normalizeOutput, 
		_dilation, 
		&minmax);
	delete[] results;
	logDebug("Height", "Height map range: " + std::to_string(minmax.x) + " to " + std::to_string(minmax.y));
}

float HeightTask::progress() const
{
	assert(_solver);
	return _solver->progress();
}
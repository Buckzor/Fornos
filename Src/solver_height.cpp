//solver_height.cpp

#include "solver_height.h"
#include "compute.h"
#include "computeshaders.h"
#include "logging.h"
#include "meshmapping.h"
#include "image.h"
#include <cassert>

static const size_t k_groupSize = 64;
static const size_t k_workPerFrame = 1024 * 128;
static const size_t k_samplePermCount = 64 * 64;

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

void HeightSolver::init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<MeshMapping> meshMapping, std::shared_ptr<MeshMapping> lowMeshMapping)
{
	_rayProgram = LoadComputeShader_Height_GenData();
	_heightProgram = LoadComputeShader_Height_Sampling();
	_avgProgram = LoadComputeShader_Height_Aggregate();

	_uvMap = map;
	_meshMapping = meshMapping;
	_lowMeshMapping = lowMeshMapping;
	_workCount = ((map->positions.size() + k_groupSize - 1) / k_groupSize) * k_groupSize;

	{
		ShaderParams params;
		params.sampleCount = (uint32_t)_params.sampleCount;
		params.samplePermCount = (uint32_t)k_samplePermCount;
		params.minDistance = _params.minDistance;
		params.maxDistance = _params.maxDistance;
		_paramsCB = std::unique_ptr<ComputeBuffer<ShaderParams> >(
			new ComputeBuffer<ShaderParams>(params, GL_STATIC_DRAW));
	}

	auto samples = computeSamples(_params.sampleCount, k_samplePermCount);
	std::vector<Vector4> samplesData(samples.begin(), samples.end());
	_samplesCB = std::unique_ptr<ComputeBuffer<Vector4> >(
		new ComputeBuffer<Vector4>(&samplesData[0], samplesData.size(), GL_STATIC_DRAW));

	_rayDataCB = std::unique_ptr<ComputeBuffer<RayData> >(
		new ComputeBuffer<RayData>(k_workPerFrame / _params.sampleCount, GL_STATIC_READ));
	_resultsMiddleCB = std::unique_ptr<ComputeBuffer<float> >(
		new ComputeBuffer<float>(k_workPerFrame, GL_STATIC_READ)); // TODO: static read?
	_resultsFinalCB = std::unique_ptr<ComputeBuffer<float> >(
		new ComputeBuffer<float>(_workCount, GL_STATIC_READ));

	_workOffset = 0;
}

bool HeightSolver::runStep()
{
	const size_t totalWork = _workCount * _params.sampleCount;
	assert(_workOffset < totalWork);
	const size_t workLeft = totalWork - _workOffset;
	const size_t work = workLeft < k_workPerFrame ? workLeft : k_workPerFrame;
	assert(work % k_groupSize == 0);

	if (_workOffset == 0) _timing.begin();

	glUseProgram(_rayProgram);
	glUniform1ui(1, GLuint(_workOffset / _params.sampleCount));
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _lowMeshMapping->meshPositions()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _lowMeshMapping->meshNormals()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _lowMeshMapping->coords()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _lowMeshMapping->coords_tidx()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _rayDataCB->bo());
	glDispatchCompute((GLuint)(work / _params.sampleCount / k_groupSize), 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glUseProgram(_heightProgram);
	glUniform1ui(1, GLuint(_workOffset / _params.sampleCount));
	glUniform1ui(2, (GLuint)_meshMapping->meshBVH()->size());
	glUniform1ui(3, (GLuint)_lowMeshMapping->meshBVH()->size());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _paramsCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _meshMapping->meshPositions()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _meshMapping->meshBVH()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, _samplesCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, _rayDataCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, _resultsMiddleCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, _lowMeshMapping->meshPositions()->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, _lowMeshMapping->meshBVH()->bo());
	glDispatchCompute((GLuint)(work / k_groupSize), 1, 1);

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glUseProgram(_avgProgram);
	glUniform1ui(1, GLuint(_workOffset / _params.sampleCount));
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _paramsCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _resultsMiddleCB->bo());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _resultsFinalCB->bo());
	glDispatchCompute((GLuint)(work / _params.sampleCount / k_groupSize), 1, 1);

	_workOffset += work;

	if (_workOffset >= totalWork)
	{
		_timing.end();
		logDebug("Height",
			"Height map took " + std::to_string(_timing.elapsedSeconds()) +
			" seconds for " + std::to_string(_uvMap->width) + "x" + std::to_string(_uvMap->height));
	}

	return _workOffset >= totalWork;
}

float* HeightSolver::getResults()
{
	//assert(_sampleIndex >= _params.sampleCount);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	return _resultsFinalCB->readData();
}

HeightTask::HeightTask(std::unique_ptr<HeightSolver> solver, const char* outputPath, int dilation)
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
	float* results = _solver->getResults();
	Vector2 minmax;
	exportFloatImage(results, _solver->uvMap().get(), _outputPath.c_str(), Vector2(0, 0), true, _dilation, &minmax);
	delete[] results;
	logDebug("Height", "Height map range: " + std::to_string(minmax.x) + " to " + std::to_string(minmax.y));
}

float HeightTask::progress() const
{
	return _solver->progress();
}
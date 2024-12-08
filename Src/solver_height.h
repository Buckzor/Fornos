//solver_height.h

#pragma once

#include "compute.h"
#include "fornos.h"
#include "math.h"
#include "timing.h"
#include <glad/glad.h>
#include <memory>
#include <vector>

struct MapUV;
class MeshMapping;

class HeightSolver
{
public:
	struct Params
	{
		size_t sampleCount;
		float minDistance;
		float maxDistance;
	};

public:
	HeightSolver(const Params &params) : _params(params) {}

	void init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<MeshMapping> meshMapping, std::shared_ptr<MeshMapping> lowMeshMapping);
	bool runStep();
	float* getResults();

	inline float progress() const
	{
		return (float)(_workOffset) / (float)(_workCount * _params.sampleCount);
	}

	inline std::shared_ptr<const CompressedMapUV> uvMap() const { return _uvMap; }

private:
	Params _params;
	size_t _workOffset;
	size_t _workCount;

	size_t _sampleIndex;

	struct ShaderParams
	{
		uint32_t sampleCount;
		uint32_t samplePermCount;
		float minDistance;
		float maxDistance;
	};

	struct RayData
	{
		Vector3 o; float _pad0;
		Vector3 d; float _pad1;
		Vector3 tx; float _pad2;
		Vector3 ty; float _pad3;
	};

	GLuint _rayProgram;
	GLuint _heightProgram;
	GLuint _avgProgram;
	std::unique_ptr<ComputeBuffer<ShaderParams> > _paramsCB;
	std::unique_ptr<ComputeBuffer<Vector4> > _samplesCB;
	std::unique_ptr<ComputeBuffer<RayData> > _rayDataCB;
	std::unique_ptr<ComputeBuffer<float> > _resultsMiddleCB;
	std::unique_ptr<ComputeBuffer<float> > _resultsFinalCB;

	std::shared_ptr<const CompressedMapUV> _uvMap;
	std::shared_ptr<MeshMapping> _meshMapping;
	std::shared_ptr<MeshMapping> _lowMeshMapping;

	Timing _timing;
};

class HeightTask : public FornosTask
{
public:
	HeightTask(std::unique_ptr<HeightSolver> solver, const char *outputPath, int dilation = 0);
	~HeightTask();

	bool runStep();
	void finish();
	float progress() const;
	const char* name() const { return "Height"; }

private:
	std::unique_ptr<HeightSolver> _solver;
	std::string _outputPath;
	int _dilation;
};

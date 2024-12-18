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

#pragma once

#include <glad/glad.h>
#include "compute.h"
#include "fornos.h"
#include "timing.h"
#include <vector>
#include <memory>

struct CompressedMapUV;
class MeshMapping;

class HeightSolver
{
public:
	struct Params
	{
		bool normalizeOutput;
		float maxDistance;
		uint32_t sampleCount = 1;   // Number of samples per pixel
		float coneAngle = 0.0f;    // Cone angle in degrees
	};

public:
	HeightSolver(const Params &params) : _params(params) {}

	void init(std::shared_ptr<const CompressedMapUV> map, std::shared_ptr<MeshMapping> mesh);
	bool runStep();
	float* getResults();

	inline float progress() const { return (float)_workOffset / (float)_workCount; }

	inline std::shared_ptr<const CompressedMapUV> uvMap() const { return _uvMap; }

	inline const Params& parameters() const { return _params; }

private:
	Params _params;
	size_t _workOffset;
	size_t _workCount;

	GLuint _heightProgram;
	std::unique_ptr<ComputeBuffer<float> > _resultsCB;
	std::unique_ptr<ComputeBuffer<Vector4> > _samplesCB; // Buffer for sample directions

	std::shared_ptr<const CompressedMapUV> _uvMap;
	std::shared_ptr<MeshMapping> _meshMapping;

	std::unique_ptr<ComputeBuffer<uint32_t>> _debugCB;

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

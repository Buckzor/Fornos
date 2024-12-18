// Auto-generated file with shaders2cpp.py utility

const char ao_step0_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
struct Output
{
vec3 o;
vec3 d;
vec3 tx;
vec3 ty;
};
layout(location = 1) uniform uint pixOffset;
layout(std430, binding = 2) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 3) readonly buffer meshNBuffer { vec3 normals[]; };
layout(std430, binding = 4) readonly buffer coordsBuffer { vec4 coords[]; };
layout(std430, binding = 5) readonly buffer coordsTidxBuffer { uint coords_tidx[]; };
layout(std430, binding = 6) writeonly buffer outputBuffer { Output outputs[]; };

vec3 getPosition(uint tidx, vec3 bcoord)
{
vec3 p0 = positions[tidx + 0];
vec3 p1 = positions[tidx + 1];
vec3 p2 = positions[tidx + 2];
return bcoord.x * p0 + bcoord.y * p1 + bcoord.z * p2;
}
vec3 getNormal(uint tidx, vec3 bcoord)
{
vec3 n0 = normals[tidx + 0];
vec3 n1 = normals[tidx + 1];
vec3 n2 = normals[tidx + 2];
return normalize(bcoord.x * n0 + bcoord.y * n1 + bcoord.z * n2);
}
void main()
{
uint in_idx = gl_GlobalInvocationID.x + pixOffset;
uint out_idx = gl_GlobalInvocationID.x;
vec4 coord = coords[in_idx];
uint tidx = coords_tidx[in_idx];
vec3 o = getPosition(tidx, coord.yzw);
vec3 d = getNormal(tidx, coord.yzw);
vec3 ty = normalize(abs(d.x) > abs(d.y) ? vec3(d.z, 0, -d.x) : vec3(0, d.z, -d.y));
vec3 tx = cross(d, ty);
outputs[out_idx].o = o;
outputs[out_idx].d = d;
outputs[out_idx].tx = tx;
outputs[out_idx].ty = ty;
}
)";

const char ao_step1_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define FLT_MAX 3.402823466e+38
struct Params
{
uint sampleCount;  
uint samplePermCount;
float minDistance;
float maxDistance;
};
struct BVH
{
float aabbMinX; float aabbMinY; float aabbMinZ;
float aabbMaxX; float aabbMaxY; float aabbMaxZ;
uint start;
uint end;
uint jump;  
};
struct Input
{
vec3 o;
vec3 d;
vec3 tx;
vec3 ty;
};
layout(location = 1) uniform uint pixOffset;
layout(location = 2) uniform uint bvhCount;
layout(std430, binding = 3) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 4) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 5) readonly buffer bvhBuffer { BVH bvhs[]; };
layout(std430, binding = 6) readonly buffer samplesBuffer { vec3 samples[]; };
layout(std430, binding = 7) readonly buffer inputsBuffer { Input inputs[]; };
layout(std430, binding = 8) writeonly buffer resultAccBuffer { float results[]; };
layout(std430, binding = 9) readonly buffer meshUVBuffer { vec2 texcoords[]; };
uniform sampler2D diffuse_tex;
float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
vec3 t1 = (mins - o) / d;
vec3 t2 = (maxs - o) / d;
vec3 tmin = min(t1, t2);
vec3 tmax = max(t1, t2);
float a = max(tmin.x, max(tmin.y, tmin.z));
float b = min(tmax.x, min(tmax.y, tmax.z));
return (b >= 0 && a <= b) ? a : FLT_MAX;
}
vec3 interplate_float3(vec3 v0, vec3 v1, vec3 v2, vec3 coeff)
{
return vec3(
dot(vec3(v0.x, v1.x, v2.x), coeff),
dot(vec3(v0.y, v1.y, v2.y), coeff),
dot(vec3(v0.z, v1.z, v2.z), coeff)
);
}
vec3 barycentric(vec3 p, vec3 a, vec3 b, vec3 c)
{
vec3 v0 = b - a;
vec3 v1 = c - a;
vec3 v2 = p - a;
float d00 = dot(v0, v0);
float d01 = dot(v0, v1);
float d11 = dot(v1, v1);
float d20 = dot(v2, v0);
float d21 = dot(v2, v1);
float denom = d00 * d11 - d01 * d01;
float y = (d11 * d20 - d01 * d21) / denom;
float z = (d00 * d21 - d01 * d20) / denom;
return vec3(1.0 - y - z, y, z);
}
 
float raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c, vec3 u0, vec3 u1, vec3 u2)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (abs(nd) > 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= 0)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
 
vec3 diffuse_uv = interplate_float3(u0, u1, u2, b);
float alpha_value = texture2D(diffuse_tex, vec2(diffuse_uv.x, 1.0f - diffuse_uv.y)).a;
if (b.x >= 0 &&  
b.y >= 0 && b.y <= 1 &&
b.z >= 0 && b.z <= 1 &&
alpha_value > 0.1)
{
return t;
}
}
}
return FLT_MAX;
}
float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist)
{
float mint = FLT_MAX;
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
vec3 u0 = vec3(texcoords[tidx + 0], 0.0f);
vec3 u1 = vec3(texcoords[tidx + 1], 0.0f);
vec3 u2 = vec3(texcoords[tidx + 2], 0.0f);
float t = raycast(o, d, v0, v1, v2, u0, u1, u2);
if (t >= mindist && t < mint)
{
mint = t;
}
}
return mint;
}
float raycastBVH(vec3 o, vec3 d, float mindist, float maxdist)
{
float mint = FLT_MAX;
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < mint && distAABB < maxdist)
 
{
float t = raycastRange(o, d, bvh.start, bvh.end, mindist);
if (t < mint)
{
mint = t;
}
++i;
}
else
{
i = bvh.jump;
}
}
return mint;
}
void main()
{
uint in_idx = gl_GlobalInvocationID.x / params.sampleCount;
uint pix_idx = in_idx + pixOffset;
uint sample_idx = gl_GlobalInvocationID.x % params.sampleCount;
uint out_idx = gl_GlobalInvocationID.x;
Input idata = inputs[in_idx];
vec3 o = idata.o;
vec3 d = idata.d;
vec3 tx = idata.tx;
vec3 ty = idata.ty;
uint sidx = (pix_idx % params.samplePermCount) * params.sampleCount + sample_idx;
vec3 rs = samples[sidx];
vec3 sampleDir = normalize(tx * rs.x + ty * rs.y + d * rs.z);
float t = raycastBVH(o, sampleDir, params.minDistance, params.maxDistance);
if (t != FLT_MAX && t < params.maxDistance)
{
results[out_idx] = 1;
}
else
{
results[out_idx] = 0;
}
}
)";

const char ao_step2_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
struct Params
{
uint sampleCount;  
float minDistance;
float maxDistance;
};
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 3) readonly buffer dataBuffer { float data[]; };
layout(std430, binding = 4) writeonly buffer resultAccBuffer { float results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x;
uint data_start_idx = gid * params.sampleCount;
float acc = 0;
for (uint i = 0; i < params.sampleCount; ++i)
{
acc += data[data_start_idx + i];
}
uint result_idx = gid + workOffset;
results[result_idx] = 1.0 - acc / float(params.sampleCount);
}
)";

const char bentnormals_step1_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define BUFFER_PARAMS 3
#define BUFFER_POSITIONS 12
#define BUFFER_BVH 8
#define BUFFER_SAMPLES 13
#define BUFFER_RESULTS_ACC 11
#define BUFFER_INPUTS 14
#define FLT_MAX 3.402823466e+38
struct Params
{
uint sampleCount;  
uint samplePermCount;
float minDistance;
float maxDistance;
};
struct BVH
{
float aabbMinX; float aabbMinY; float aabbMinZ;
float aabbMaxX; float aabbMaxY; float aabbMaxZ;
uint start;
uint end;
uint jump;  
};
struct Input
{
vec3 o;
vec3 d;
vec3 tx;
vec3 ty;
};
layout(location = 1) uniform uint pixOffset;
layout(location = 2) uniform uint bvhCount;
layout(std430, binding = 3) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 4) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 5) readonly buffer bvhBuffer { BVH bvhs[]; };
layout(std430, binding = 6) readonly buffer samplesBuffer { vec3 samples[]; };
layout(std430, binding = 7) readonly buffer inputsBuffer { Input inputs[]; };
layout(std430, binding = 8) writeonly buffer resultAccBuffer { vec3 results[]; };
float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
vec3 t1 = (mins - o) / d;
vec3 t2 = (maxs - o) / d;
vec3 tmin = min(t1, t2);
vec3 tmax = max(t1, t2);
float a = max(tmin.x, max(tmin.y, tmin.z));
float b = min(tmax.x, min(tmax.y, tmax.z));
return (b >= 0 && a <= b) ? a : FLT_MAX;
}
vec3 barycentric(vec3 p, vec3 a, vec3 b, vec3 c)
{
    vec3 v0 = b - a;
    vec3 v1 = c - a;
    vec3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float y = (d11 * d20 - d01 * d21) / denom;
    float z = (d00 * d21 - d01 * d20) / denom;
    return vec3(1.0 - y - z, y, z);
}
 
float raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (abs(nd) > 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= 0)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
if (b.x >= 0 &&  
b.y >= 0 && b.y <= 1 &&
b.z >= 0 && b.z <= 1)
{
return t;
}
}
}
return FLT_MAX;
}
float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist)
{
float mint = FLT_MAX;
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
float t = raycast(o, d, v0, v1, v2);
if (t >= mindist && t < mint)
{
mint = t;
}
}
return mint;
}
float raycastBVH(vec3 o, vec3 d, float mindist, float maxdist)
{
float mint = FLT_MAX;
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < mint && distAABB < maxdist)
 
{
float t = raycastRange(o, d, bvh.start, bvh.end, mindist);
if (t < mint)
{
mint = t;
}
++i;
}
else
{
i = bvh.jump;
}
}
return mint;
}
void main()
{
uint in_idx = gl_GlobalInvocationID.x / params.sampleCount;
uint pix_idx = in_idx + pixOffset;
uint sample_idx = gl_GlobalInvocationID.x % params.sampleCount;
uint out_idx = gl_GlobalInvocationID.x;
Input idata = inputs[in_idx];
vec3 o = idata.o;
vec3 d = idata.d;
vec3 tx = idata.tx;
vec3 ty = idata.ty;
uint sidx = (pix_idx % params.samplePermCount) * params.sampleCount + sample_idx;
vec3 rs = samples[sidx]*0.0;
vec3 sampleDir = normalize(tx * rs.x + ty * rs.y + d * rs.z);
float t = raycastBVH(o, sampleDir, params.minDistance, params.maxDistance);
results[out_idx] = (t != FLT_MAX) ? vec3(0,0,0) : sampleDir;
}
)";

const char bentnormals_step2_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
struct Params
{
uint sampleCount;  
float minDistance;
float maxDistance;
};
struct V3 { float x; float y; float z; };
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 3) readonly buffer dataBuffer { vec3 data[]; };
layout(std430, binding = 4) writeonly buffer resultAccBuffer { V3 results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x;
uint data_start_idx = gid * params.sampleCount;
vec3 acc = vec3(0, 0, 0);
for (uint i = 0; i < params.sampleCount; ++i)
{
acc += data[data_start_idx + i];
}
vec3 normal = normalize(acc);
uint result_idx = gid + workOffset;
results[result_idx].x = normal.x;
results[result_idx].y = normal.y;
results[result_idx].z = normal.z;
}
)";
const char height_step0_comp[] =
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout (local_size_x = 64) in;

struct Output {
    vec3 o;  // Ray origin
    vec3 d;  // Ray direction
    vec3 tx; // Tangent direction for sampling
    vec3 ty; // Bitangent direction for sampling
};

layout(location = 1) uniform uint pixOffset;
layout(std430, binding = 2) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 3) readonly buffer meshNBuffer { vec3 normals[]; };
layout(std430, binding = 4) readonly buffer coordsBuffer { vec4 coords[]; };
layout(std430, binding = 5) readonly buffer coordsTidxBuffer { uint coords_tidx[]; };
layout(std430, binding = 6) writeonly buffer outputBuffer { Output outputs[]; };

vec3 getPosition(uint tidx, vec3 bcoord) {
    vec3 p0 = positions[tidx + 0];
    vec3 p1 = positions[tidx + 1];
    vec3 p2 = positions[tidx + 2];
    return bcoord.x * p0 + bcoord.y * p1 + bcoord.z * p2;
}

vec3 getNormal(uint tidx, vec3 bcoord) {
    vec3 n0 = normals[tidx + 0];
    vec3 n1 = normals[tidx + 1];
    vec3 n2 = normals[tidx + 2];
    return normalize(bcoord.x * n0 + bcoord.y * n1 + bcoord.z * n2);
}

void main() {
    uint in_idx = gl_GlobalInvocationID.x + pixOffset;
    uint out_idx = gl_GlobalInvocationID.x;

    vec4 coord = coords[in_idx];
    uint tidx = coords_tidx[in_idx];

    // Calculate ray origin and normal direction from the low-poly mesh
    vec3 o = getPosition(tidx, coord.yzw);
    vec3 d = getNormal(tidx, coord.yzw); // Ray direction follows the normal direction (inwards)

    // Increase the small offset along the opposite of the normal to ensure the ray starts slightly inside the mesh
    const float epsilon = 0.01;  // Offset to start the ray just inside the mesh surface
    vec3 rayOrigin = o - epsilon * d;

    // Calculate tangent and bitangent vectors
    vec3 ty = normalize(abs(d.x) > abs(d.y) ? vec3(d.z, 0, -d.x) : vec3(0, d.z, -d.y));
    vec3 tx = normalize(cross(d, ty));

    // Write output
    outputs[out_idx].o = rayOrigin;  // Use the adjusted ray origin (slightly inside)
    outputs[out_idx].d = -d;         // Invert direction to point towards the mesh interior
    outputs[out_idx].tx = tx;
    outputs[out_idx].ty = ty;
}
)";

const char height_step1_comp[] =
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout (local_size_x = 64) in;

#define FLT_MAX 3.402823466e+38

struct Params {
    uint sampleCount;  
    uint samplePermCount;
    float minDistance;
    float maxDistance;
};

struct BVH {
    float aabbMinX; float aabbMinY; float aabbMinZ;
    float aabbMaxX; float aabbMaxY; float aabbMaxZ;
    uint start;
    uint end;
    uint jump;  
};

struct Input {
    vec3 o;
    vec3 d;
    vec3 tx;
    vec3 ty;
};

layout(location = 1) uniform uint pixOffset;
layout(location = 2) uniform uint highBvhCount;
layout(location = 3) uniform uint lowBvhCount;

layout(std430, binding = 3) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 4) readonly buffer highMeshPBuffer { vec3 highPositions[]; };
layout(std430, binding = 5) readonly buffer highBvhBuffer { BVH highBvhs[]; };
layout(std430, binding = 6) readonly buffer samplesBuffer { vec3 samples[]; };
layout(std430, binding = 7) readonly buffer inputsBuffer { Input inputs[]; };
layout(std430, binding = 8) writeonly buffer resultAccBuffer { float results[]; };
layout(std430, binding = 9) readonly buffer lowMeshPBuffer { vec3 lowPositions[]; };
layout(std430, binding = 10) readonly buffer lowBvhBuffer { BVH lowBvhs[]; };

// Barycentric function to determine the barycentric coordinates of point p
vec3 barycentric(vec3 p, vec3 a, vec3 b, vec3 c)
{
    vec3 v0 = b - a;
    vec3 v1 = c - a;
    vec3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float y = (d11 * d20 - d01 * d21) / denom;
    float z = (d00 * d21 - d01 * d20) / denom;
    return vec3(1.0 - y - z, y, z);
}

float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
    vec3 t1 = (mins - o) / d;
    vec3 t2 = (maxs - o) / d;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    float a = max(tmin.x, max(tmin.y, tmin.z));
    float b = min(tmax.x, min(tmax.y, tmax.z));
    return (b >= 0 && a <= b) ? a : FLT_MAX;
}

float raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
    vec3 n = normalize(cross(b - a, c - a));
    float nd = dot(d, n);
    if (abs(nd) > 0.0001)  // Adding a small epsilon to avoid numerical issues
    {
        float pn = dot(o, n);
        float t = (dot(a, n) - pn) / nd;
        const float bias = 0.01; // Increase bias to avoid self-intersection
        if (t >= bias)  // Ensure we skip very close intersections to origin
        {
            vec3 p = o + d * t;
            vec3 bCoord = barycentric(p, a, b, c);
            if (bCoord.x >= 0 && bCoord.y >= 0 && bCoord.y <= 1 && bCoord.z >= 0 && bCoord.z <= 1)
            {
                return t;
            }
        }
    }
    return FLT_MAX;
}

float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist, bool useHighMesh)
{
    float mint = FLT_MAX;
    for (uint tidx = start; tidx < end; tidx += 3)
    {
        vec3 v0 = useHighMesh ? highPositions[tidx + 0] : lowPositions[tidx + 0];
        vec3 v1 = useHighMesh ? highPositions[tidx + 1] : lowPositions[tidx + 1];
        vec3 v2 = useHighMesh ? highPositions[tidx + 2] : lowPositions[tidx + 2];
        float t = raycast(o, d, v0, v1, v2);
        if (t >= mindist && t < mint)
        {
            mint = t;
        }
    }
    return mint;
}

float raycastBVH(vec3 o, vec3 d, float mindist, float maxdist, bool useHighMesh)
{
    float mint = FLT_MAX;
    uint i = 0;
    uint bvhCount = useHighMesh ? highBvhCount : lowBvhCount;

    while (i < bvhCount)
    {
        BVH bvh = useHighMesh ? highBvhs[i] : lowBvhs[i];
        vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
        vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
        float distAABB = RayAABB(o, d, aabbMin, aabbMax);
        if (distAABB < mint && distAABB < maxdist)
        {
            float t = raycastRange(o, d, bvh.start, bvh.end, mindist, useHighMesh);
            if (t < mint)
            {
                mint = t;
            }
            ++i;
        }
        else
        {
            i = bvh.jump;
        }
    }
    return mint;
}

void main() {
    uint in_idx = gl_GlobalInvocationID.x / params.sampleCount;
    uint pix_idx = in_idx + pixOffset;
    uint sample_idx = gl_GlobalInvocationID.x % params.sampleCount;
    uint out_idx = gl_GlobalInvocationID.x;

    Input idata = inputs[in_idx];
    vec3 o = idata.o;  // Starting point on low poly model
    vec3 d = idata.d;  // Ray direction (opposite to surface normal)

    // Calculate the ray direction for sampling (towards the high poly model)
    vec3 tx = idata.tx;
    vec3 ty = idata.ty;
    uint sidx = (pix_idx % params.samplePermCount) * params.sampleCount + sample_idx;
    vec3 rs = samples[sidx];
    vec3 sampleDir = normalize(tx * rs.x + ty * rs.y + d * rs.z);
    sampleDir = mix(sampleDir,d,params.minDistance); // Reducing the sampling spread to avoid excessive blurriness

    // Cast the ray to intersect with the high poly mesh (skeleton)
    float tHigh = raycastBVH(o, sampleDir, 0.0, params.maxDistance, true);

    // If no high poly intersection, proceed to check for low poly backface intersection
    if (tHigh == FLT_MAX) {
        // Check intersection with low poly backfaces
        float tLow = raycastBVH(o, sampleDir, 0.0, params.maxDistance, false);
        if (tLow != FLT_MAX) {
            // Set the result by inverting the distance to make closer values white
            results[out_idx] = clamp(1.0 - (tLow / params.maxDistance), 0.0, 1.0);
        } else {
            // Set to black (0.0) for no intersection, implying farthest distance.
            results[out_idx] = 0.0;
        }
    } else {
        // Intersection with high poly mesh
        // Set the result by inverting the distance to make closer values white
        results[out_idx] = clamp(1.0 - (tHigh / params.maxDistance), 0.0, 1.0);
    }
}
)";
const char height_step2_comp[] =
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
struct Params
{
    uint sampleCount;  
    float minDistance;
    float maxDistance;
};
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 3) readonly buffer dataBuffer { float data[]; };
layout(std430, binding = 4) writeonly buffer resultAccBuffer { float results[]; };

void main()
{
    uint gid = gl_GlobalInvocationID.x;
    uint data_start_idx = gid * params.sampleCount;
    float acc = 0.0;

    for (uint i = 0; i < params.sampleCount; ++i)
    {
        acc += data[data_start_idx + i];
    }

    uint result_idx = gid + workOffset;
    results[result_idx] = acc / float(params.sampleCount);
}
)";
const char meshmapping_comp[] =
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define RAYCAST_FORWARD 1
#define RAYCAST_BACKWARD 1
#define FLT_MAX 3.402823466e+38
#define BARY_MIN -1e-5
#define BARY_MAX 1.0
struct Pix
{
vec3 p;
vec3 d;
};
struct BVH
{
float aabbMinX; float aabbMinY; float aabbMinZ;
float aabbMaxX; float aabbMaxY; float aabbMaxZ;
uint start;
uint end;
uint jump;  
};
layout(location = 1) uniform uint workOffset;
layout(location = 2) uniform uint workCount;
layout(location = 3) uniform uint bvhCount;
layout(std430, binding = 4) readonly buffer pixBuffer { Pix pixels[]; };
layout(std430, binding = 5) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 6) readonly buffer bvhBuffer { BVH bvhs[]; };
layout(std430, binding = 7) writeonly buffer rCoordBuffer { vec4 r_coords[]; };
layout(std430, binding = 8) writeonly buffer rTidxBuffer { uint r_tidx[]; };
float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
vec3 dabs = abs(d);
vec3 t1 = (mins - o) / d;
vec3 t2 = (maxs - o) / d;
vec3 tmin = min(t1, t2);
vec3 tmax = max(t1, t2);
float a = max(tmin.x, max(tmin.y, tmin.z));
float b = min(tmax.x, min(tmax.y, tmax.z));
return (b >= 0 && a <= b) ? a : FLT_MAX;
}
vec3 barycentric(dvec3 p, dvec3 a, dvec3 b, dvec3 c)
{
dvec3 v0 = b - a;
dvec3 v1 = c - a;
dvec3 v2 = p - a;
double d00 = dot(v0, v0);
double d01 = dot(v0, v1);
double d11 = dot(v1, v1);
double d20 = dot(v2, v0);
double d21 = dot(v2, v1);
double denom = d00 * d11 - d01 * d01;
double y = (d11 * d20 - d01 * d21) / denom;
double z = (d00 * d21 - d01 * d20) / denom;
return vec3(dvec3(1.0 - y - z, y, z));
}
 
vec4 raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (abs(nd) > 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= 0)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
if (b.x >= BARY_MIN && b.y >= BARY_MIN && b.y <= BARY_MAX && b.z >= BARY_MIN && b.z <= BARY_MAX)
{
return vec4(t, b.x, b.y, b.z);
}
}
}
return vec4(FLT_MAX, 0, 0, 0);
}
float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist, out uint o_idx, out vec3 o_bcoord)
{
float mint = FLT_MAX;
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
vec4 r = raycast(o, d, v0, v1, v2);
if (r.x >= mindist && r.x < mint)
{
mint = r.x;
o_idx = tidx;
o_bcoord = r.yzw;
}
}
return mint;
}
float raycastBVH(vec3 o, vec3 d, float mint, in out uint o_idx, in out vec3 o_bcoord)
{
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < mint)
 
{
uint ridx = 0;
vec3 rbcoord = vec3(0, 0, 0);
float t = raycastRange(o, d, bvh.start, bvh.end, 0, ridx, rbcoord);
if (t < mint)
{
mint = t;
o_idx = ridx;
o_bcoord = rbcoord;
}
++i;
}
else
{
i = bvh.jump;
}
}
return mint;
}
void main()
{
uint gid = gl_GlobalInvocationID.x + workOffset;
if (gid >= workCount) return;
Pix pix = pixels[gid];
vec3 p = pix.p;
vec3 d = pix.d;
uint tidx = 4294967295;
vec3 bcoord = vec3(0, 0, 0);
float t = FLT_MAX;
#if RAYCAST_FORWARD
t = min(t, raycastBVH(p, d, t, tidx, bcoord));
#endif
#if RAYCAST_BACKWARD
t = min(t, raycastBVH(p, -d, t, tidx, bcoord));
#endif
r_coords[gid] = vec4(t, bcoord.x, bcoord.y, bcoord.z);
r_tidx[gid] = tidx;
}
)";
const char meshmapping_nobackfaces_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
#extension GL_ARB_gpu_shader_fp64 : enable
layout (local_size_x = 64) in;
#define FLT_MAX 3.402823466e+38
#define BARY_MIN -1e-5
#define BARY_MAX 1.0
struct Pix
{
vec3 p;
vec3 d;
};
struct BVH
{
float aabbMinX; float aabbMinY; float aabbMinZ;
float aabbMaxX; float aabbMaxY; float aabbMaxZ;
uint start;
uint end;
uint jump;  
};
layout(location = 1) uniform uint workOffset;
layout(location = 2) uniform uint workCount;
layout(location = 3) uniform uint bvhCount;
layout(std430, binding = 4) readonly buffer pixBuffer { Pix pixels[]; };
layout(std430, binding = 5) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 6) readonly buffer bvhBuffer { BVH bvhs[]; };
layout(std430, binding = 7) writeonly buffer rCoordBuffer { vec4 r_coords[]; };
layout(std430, binding = 8) writeonly buffer rTidxBuffer { uint r_tidx[]; };
float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
vec3 dabs = abs(d);
vec3 t1 = (mins - o) / d;
vec3 t2 = (maxs - o) / d;
vec3 tmin = min(t1, t2);
vec3 tmax = max(t1, t2);
float a = max(tmin.x, max(tmin.y, tmin.z));
float b = min(tmax.x, min(tmax.y, tmax.z));
return (b >= 0 && a <= b) ? a : FLT_MAX;
}
vec3 barycentric(dvec3 p, dvec3 a, dvec3 b, dvec3 c)
{
dvec3 v0 = b - a;
dvec3 v1 = c - a;
dvec3 v2 = p - a;
double d00 = dot(v0, v0);
double d01 = dot(v0, v1);
double d11 = dot(v1, v1);
double d20 = dot(v2, v0);
double d21 = dot(v2, v1);
double denom = d00 * d11 - d01 * d01;
double y = (d11 * d20 - d01 * d21) / denom;
double z = (d00 * d21 - d01 * d20) / denom;
return vec3(dvec3(1.0 - y - z, y, z));
}
vec4 raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c, float mindist, float maxdist)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (nd > 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= mindist && t < maxdist)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
if (b.x >= BARY_MIN && b.y >= BARY_MIN && b.y <= BARY_MAX && b.z >= BARY_MIN && b.z <= BARY_MAX)
{
return vec4(t, b.x, b.y, b.z);
}
}
}
return vec4(FLT_MAX, 0, 0, 0);
}
void raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist, in out float curdist, in out uint o_idx, in out vec3 o_bcoord)
{
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
vec4 r = raycast(o, d, v0, v1, v2, mindist, curdist);
if (r.x != FLT_MAX)
{
curdist = r.x;
o_idx = tidx;
o_bcoord = r.yzw;
}
}
}
void raycastBVH(vec3 o, vec3 d, in out float curdist, in out uint o_idx, in out vec3 o_bcoord)
{
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < curdist)
{
raycastRange(o, d, bvh.start, bvh.end, 0, curdist, o_idx, o_bcoord);
++i;
}
else
{
i = bvh.jump;
}
}
}
vec4 raycastBack(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c, float mindist, float maxdist)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (nd < 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= mindist && t < maxdist)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
if (b.x >= BARY_MIN && b.y >= BARY_MIN && b.y <= BARY_MAX && b.z >= BARY_MIN && b.z <= BARY_MAX)
{
return vec4(t, b.x, b.y, b.z);
}
}
}
return vec4(FLT_MAX, 0, 0, 0);
}
void raycastBackRange(vec3 o, vec3 d, uint start, uint end, float mindist, in out float curdist, in out uint o_idx, in out vec3 o_bcoord)
{
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
vec4 r = raycastBack(o, d, v0, v1, v2, mindist, curdist);
if (r.x != FLT_MAX)
{
curdist = r.x;
o_idx = tidx;
o_bcoord = r.yzw;
}
}
}
void raycastBackBVH(vec3 o, vec3 d, in out float curdist, in out uint o_idx, in out vec3 o_bcoord)
{
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < curdist)
{
raycastBackRange(o, d, bvh.start, bvh.end, 0, curdist, o_idx, o_bcoord);
++i;
}
else
{
i = bvh.jump;
}
}
}
void main()
{
uint gid = gl_GlobalInvocationID.x + workOffset;
if (gid >= workCount) return;
Pix pix = pixels[gid];
vec3 p = pix.p;
vec3 d = pix.d;
uint tidx = 4294967295;
vec3 bcoord = vec3(0, 0, 0);
float t = FLT_MAX;
raycastBVH(p, d, t, tidx, bcoord);
raycastBackBVH(p, -d, t, tidx, bcoord);
r_coords[gid] = vec4(t, bcoord.x, bcoord.y, bcoord.z);
r_tidx[gid] = tidx;
}
)";
const char normals_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer meshNBuffer { vec3 normals[]; };
layout(std430, binding = 3) readonly buffer coordsBuffer { vec4 coords[]; };
layout(std430, binding = 4) readonly buffer coordsTidxBuffer { uint coords_tidx[]; };
layout(std430, binding = 5) writeonly buffer resultBuffer { float results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x + workOffset;
vec4 coord = coords[gid];
uint tidx = coords_tidx[gid];
vec3 n0 = normals[tidx + 0];
vec3 n1 = normals[tidx + 1];
vec3 n2 = normals[tidx + 2];
vec3 normal = normalize(coord.y * n0 + coord.z * n1 + coord.w * n2);
uint ridx = gid * 3;
results[ridx + 0] = normal.x;
results[ridx + 1] = normal.y;
results[ridx + 2] = normal.z;
}
)";
const char positions_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define FLT_MAX 3.402823466e+38
#define TANGENT_SPACE 0
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 3) readonly buffer coordsBuffer { vec4 coords[]; };
layout(std430, binding = 4) readonly buffer coordsTidxBuffer { uint coords_tidx[]; };
layout(std430, binding = 5) writeonly buffer resultBuffer { float results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x + workOffset;
vec4 coord = coords[gid];
uint tidx = coords_tidx[gid];
vec3 p0 = positions[tidx + 0];
vec3 p1 = positions[tidx + 1];
vec3 p2 = positions[tidx + 2];
vec3 p = coord.y * p0 + coord.z * p1 + coord.w * p2;
uint ridx = gid * 3;
results[ridx + 0] = p.x;
results[ridx + 1] = p.y;
results[ridx + 2] = p.z;
}
)";
const char tangentspace_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define TANGENT_SPACE 1
struct PixelT
{
vec3 n;
vec3 t;
vec3 b;
};
struct V3 { float x; float y; float z; };
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer pixtBuffer { PixelT pixelst[]; };
layout(std430, binding = 3) buffer resultBuffer { V3 results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x;
uint result_idx = gid + workOffset;
vec3 normal = vec3(results[result_idx].x, results[result_idx].y, results[result_idx].z);
PixelT pixt = pixelst[result_idx];
vec3 n = pixt.n;
vec3 t = pixt.t;
vec3 b = pixt.b;
vec3 d0 = vec3(n.z*b.y - n.y*b.z, n.x*b.z - n.z*b.x, n.y*b.x - n.x*b.y);
vec3 d1 = vec3(t.z*n.y - t.y*n.z, t.x*n.z - n.x*t.z, n.x*t.y - t.x*n.y);
vec3 d2 = vec3(t.y*b.z - t.z*b.y, t.z*b.x - t.x*b.z, t.x*b.y - t.y*b.x);
normal = normalize(vec3(dot(normal, d0), dot(normal, d1), dot(normal, d2)));
results[result_idx].x = normal.x;
results[result_idx].y = normal.y;
results[result_idx].z = normal.z;
}
)";
const char thick_step1_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
#define FLT_MAX 3.402823466e+38
struct Params
{
uint sampleCount;  
uint samplePermCount;
float minDistance;
float maxDistance;
};
struct BVH
{
float aabbMinX; float aabbMinY; float aabbMinZ;
float aabbMaxX; float aabbMaxY; float aabbMaxZ;
uint start;
uint end;
uint jump;  
};
struct Input
{
vec3 o;
vec3 d;
vec3 tx;
vec3 ty;
};
layout(location = 1) uniform uint pixOffset;
layout(location = 2) uniform uint bvhCount;
layout(std430, binding = 3) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 4) readonly buffer meshPBuffer { vec3 positions[]; };
layout(std430, binding = 5) readonly buffer bvhBuffer { BVH bvhs[]; };
layout(std430, binding = 6) readonly buffer samplesBuffer { vec3 samples[]; };
layout(std430, binding = 7) readonly buffer inputsBuffer { Input inputs[]; };
layout(std430, binding = 8) writeonly buffer resultAccBuffer { float results[]; };
float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
vec3 t1 = (mins - o) / d;
vec3 t2 = (maxs - o) / d;
vec3 tmin = min(t1, t2);
vec3 tmax = max(t1, t2);
float a = max(tmin.x, max(tmin.y, tmin.z));
float b = min(tmax.x, min(tmax.y, tmax.z));
return (b >= 0 && a <= b) ? a : FLT_MAX;
}
vec3 barycentric(vec3 p, vec3 a, vec3 b, vec3 c)
{
vec3 v0 = b - a;
vec3 v1 = c - a;
vec3 v2 = p - a;
float d00 = dot(v0, v0);
float d01 = dot(v0, v1);
float d11 = dot(v1, v1);
float d20 = dot(v2, v0);
float d21 = dot(v2, v1);
float denom = d00 * d11 - d01 * d01;
float y = (d11 * d20 - d01 * d21) / denom;
float z = (d00 * d21 - d01 * d20) / denom;
return vec3(1.0 - y - z, y, z);
}
 
float raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
vec3 n = normalize(cross(b - a, c - a));
float nd = dot(d, n);
if (abs(nd) > 0)
{
float pn = dot(o, n);
float t = (dot(a, n) - pn) / nd;
if (t >= 0)
{
vec3 p = o + d * t;
vec3 b = barycentric(p, a, b, c);
if (b.x >= 0 &&  
b.y >= 0 && b.y <= 1 &&
b.z >= 0 && b.z <= 1)
{
return t;
}
}
}
return FLT_MAX;
}
float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist)
{
float mint = FLT_MAX;
for (uint tidx = start; tidx < end; tidx += 3)
{
vec3 v0 = positions[tidx + 0];
vec3 v1 = positions[tidx + 1];
vec3 v2 = positions[tidx + 2];
float t = raycast(o, d, v0, v1, v2);
if (t >= mindist && t < mint)
{
mint = t;
}
}
return mint;
}
float raycastBVH(vec3 o, vec3 d, float mindist, float maxdist)
{
float mint = FLT_MAX;
uint i = 0;
while (i < bvhCount)
{
BVH bvh = bvhs[i];
vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
float distAABB = RayAABB(o, d, aabbMin, aabbMax);
if (distAABB < mint && distAABB < maxdist)
 
{
float t = raycastRange(o, d, bvh.start, bvh.end, mindist);
if (t < mint)
{
mint = t;
}
++i;
}
else
{
i = bvh.jump;
}
}
return mint;
}
void main()
{
uint in_idx = gl_GlobalInvocationID.x / params.sampleCount;
uint pix_idx = in_idx + pixOffset;
uint sample_idx = gl_GlobalInvocationID.x % params.sampleCount;
uint out_idx = gl_GlobalInvocationID.x;
Input idata = inputs[in_idx];
vec3 o = idata.o;
vec3 d = -idata.d;
vec3 tx = idata.tx;
vec3 ty = idata.ty;
uint sidx = (pix_idx % params.samplePermCount) * params.sampleCount + sample_idx;
vec3 rs = samples[sidx];
vec3 sampleDir = normalize(tx * rs.x + ty * rs.y + d * rs.z);
float t = raycastBVH(o, sampleDir, params.minDistance, params.maxDistance);
results[out_idx] = (t != FLT_MAX) ? t : params.maxDistance;
}
)";
const char thick_step2_comp[] = 
R"(
#version 430 core
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable
layout (local_size_x = 64) in;
struct Params
{
uint sampleCount;  
float minDistance;
float maxDistance;
};
layout(location = 1) uniform uint workOffset;
layout(std430, binding = 2) readonly buffer paramsBuffer { Params params; };
layout(std430, binding = 3) readonly buffer dataBuffer { float data[]; };
layout(std430, binding = 4) writeonly buffer resultAccBuffer { float results[]; };
void main()
{
uint gid = gl_GlobalInvocationID.x;
uint data_start_idx = gid * params.sampleCount;
float acc = 0;
for (uint i = 0; i < params.sampleCount; ++i)
{
acc += data[data_start_idx + i];
}
uint result_idx = gid + workOffset;
results[result_idx] = acc / float(params.sampleCount);
}
)";


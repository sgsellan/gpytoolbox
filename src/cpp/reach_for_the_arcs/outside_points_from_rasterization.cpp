#include "outside_points_from_rasterization.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include <random>
#include <iostream>
#include <algorithm>
#include <igl/parallel_for.h>


#ifdef GL_AVAILABLE

// All GL code follows the tutorial https://eliemichel.github.io/LearnWebGPU
//#include <GLFW/glfw3.h>
//#include <glfw3webgpu.h>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

const char* shader_src = R"(
struct Uniforms {
    res : f32,
    narrow_band_by : f32,
    z : f32,
    render_pass : i32
};
@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexInput {
    @location(0) c : vec3f,
    @location(1) r : f32
};

struct VertexOutput {
    @builtin(position) position : vec4f,
    @location(0) c : vec2f,
    @location(1) r : f32,
    @location(2) banded_r : f32
};

fn ssqrt(x : f32) -> f32
{
    return sqrt(max(x, 0.));
}

fn norm(v : vec2f) -> f32
{
    return ssqrt(v.x*v.x + v.y*v.y);
}

@vertex
fn vs_main(
    in : VertexInput,
    @builtin(vertex_index) idx : u32
    ) -> VertexOutput {

    var quad_pos : vec2f;
    if(idx == 0u) {
        quad_pos = 2.*vec2f(0.,0.) - 1.;
    } else if(idx==1u) {
        quad_pos = 2.*vec2f(1.,0.) - 1.;
    } else if(idx==2u) {
        quad_pos = 2.*vec2f(0.,1.) - 1.;
    } else if(idx==3u) {
        quad_pos = 2.*vec2f(0.,1.) - 1.;
    } else if(idx==4u) {
        quad_pos = 2.*vec2f(1.,0.) - 1.;
    } else if(idx==5u) {
        quad_pos = 2.*vec2f(1.,1.) - 1.;
    }

    let pos = vec2f(in.c.x, in.c.y);
    let zdiff = in.c.z - u.z;
    let r = ssqrt(in.r*in.r - zdiff*zdiff);
    let in_r_plus_band = in.r + u.narrow_band_by;
    let banded_r = ssqrt(in_r_plus_band*in_r_plus_band - zdiff*zdiff);
    var out : VertexOutput;
    out.position = vec4f(pos +  banded_r*quad_pos, 0.0, 1.0);
    out.c = pos;
    out.r = r;
    out.banded_r = banded_r;
    return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4f {
    var pos = 2.*in.position.xy/u.res - 1.;
    pos.y = -pos.y;
    let d_to_c = norm(pos-in.c);

    if(u.narrow_band_by > 0.0) {
        if(u.render_pass == 0) {
            let cond = f32(d_to_c < in.banded_r);
            return vec4f(0.0, 0.0, 0.0, cond);
        } else { //u.render_pass == 1
            let cond = f32(d_to_c < in.r);
            return vec4f(cond, cond, cond, cond);
        }
    } else {
        let cond = f32(d_to_c < in.r);
        return vec4f(cond, cond, cond, cond);
    }
}
)";

struct Uniforms {
    float res;
    float narrow_band_by;
    float z;
    int32_t pass;
};

template<int dim>
bool outside_points_from_rasterization_gpu(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    int res,
    const double tol,
    const bool narrow_band,
    const bool parallel,
    const bool verbose,
    Eigen::MatrixXd & outside_points)
{
    using Vecd = Eigen::Matrix<double,dim,1>;

    const double raster_shrink = 0.95;

    // For wgpu, res must be divisible by 64.
    if(res%64 != 0) {
        res = res + 64 - (res%64);
    }

    // wgpu/GL boilerplate setup.
    using namespace wgpu;
    Instance instance = createInstance(InstanceDescriptor{});
    if (!instance) {
        if(verbose) {
            std::cout << "Error initializing WebGPU." << std::endl;
        }
        return false;
    }
    // if(!glfwInit()) {
    //     if(verbose) {
    //         std::cout << "Error initializing GLFW." << std::endl;
    //     }
    //     return false;
    // }
    // glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // GLFWwindow* window = glfwCreateWindow(res, res, "Rasterization", NULL, NULL);
    // if(!window) {
    //     if(verbose) {
    //         std::cout << "Error opening GLFW window." << std::endl;
    //     }
    //     glfwTerminate();
    //     return false;
    // }
    // Surface surface = glfwGetWGPUSurface(instance, window);
    RequestAdapterOptions adapterOpts;
    adapterOpts.compatibleSurface = nullptr; //surface;
    Adapter adapter = instance.requestAdapter(adapterOpts);
    SupportedLimits supportedLimits;
    adapter.getLimits(&supportedLimits);
    RequiredLimits requiredLimits = Default;
    requiredLimits.limits.maxVertexAttributes = 2;
    requiredLimits.limits.maxVertexBuffers = 1;
    requiredLimits.limits.maxBufferSize = std::max(
        sdf_points.rows() * 4 * sizeof(float),
        res * res * sizeof(uint32_t));
    requiredLimits.limits.maxVertexBufferArrayStride = 4 * sizeof(float);
    requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;
    requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
    requiredLimits.limits.maxInterStageShaderComponents = 4;
    requiredLimits.limits.maxBindGroups = 1;
    requiredLimits.limits.maxUniformBuffersPerShaderStage = 1;
    requiredLimits.limits.maxUniformBufferBindingSize = 2 * sizeof(Uniforms);
    requiredLimits.limits.maxTextureDimension1D = res;
    requiredLimits.limits.maxTextureDimension2D = res;
    requiredLimits.limits.maxTextureDimension3D = res;
    requiredLimits.limits.maxTextureArrayLayers = 1;
    DeviceDescriptor deviceDesc;
    deviceDesc.label = "Device";
    deviceDesc.requiredFeaturesCount = 0;
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.label = "Queue";
    Device device = adapter.requestDevice(deviceDesc);
    Queue queue = device.getQueue();
    wgpuDeviceSetUncapturedErrorCallback(device, [](WGPUErrorType type,
        char const* message, void*) {
        std::cout << "Device error " << type;
        if(message) {
            std::cout << " (" << message << ")" << std::endl;
        }
    }, nullptr);
    TextureDescriptor textureDesc;
    textureDesc.dimension = TextureDimension::_2D;
    textureDesc.format = TextureFormat::RGBA8Unorm;
    textureDesc.sampleCount = 1;
    textureDesc.mipLevelCount = 1;
    textureDesc.label = "Texture";
    textureDesc.usage = TextureUsage::CopySrc | TextureUsage::RenderAttachment;
    textureDesc.nextInChain = nullptr;
    textureDesc.size = {
        static_cast<uint32_t>(res), static_cast<uint32_t>(res), 1};
    textureDesc.viewFormatCount = 1;
    textureDesc.viewFormats = (WGPUTextureFormat*)&textureDesc.format;
    Texture texture = device.createTexture(textureDesc);
    TextureViewDescriptor textureViewDesc;
    textureViewDesc.aspect = TextureAspect::All;
    textureViewDesc.baseArrayLayer = 0;
    textureViewDesc.arrayLayerCount = 1;
    textureViewDesc.baseMipLevel = 0;
    textureViewDesc.mipLevelCount = 1;
    textureViewDesc.dimension = TextureViewDimension::_2D;
    textureViewDesc.format = textureDesc.format;
    TextureView textureView = texture.createView(textureViewDesc);
    // TextureFormat swapChainFormat = surface.getPreferredFormat(adapter);
    // SwapChainDescriptor swapChainDesc = {};
    // swapChainDesc.width = res;
    // swapChainDesc.height = res;
    // swapChainDesc.usage = TextureUsage::RenderAttachment;
    // swapChainDesc.format = swapChainFormat;
    // swapChainDesc.presentMode = PresentMode::Fifo;
    // SwapChain swapChain = device.createSwapChain(surface, swapChainDesc);
    ShaderModuleDescriptor shaderDesc;
    shaderDesc.hintCount = 0;
    shaderDesc.hints = nullptr;
    ShaderModuleWGSLDescriptor shaderCodeDesc;
    shaderCodeDesc.chain.next = nullptr;
    shaderCodeDesc.chain.sType = SType::ShaderModuleWGSLDescriptor;
    shaderDesc.nextInChain = &shaderCodeDesc.chain;
    shaderCodeDesc.code = shader_src;
    ShaderModule shaderModule = device.createShaderModule(shaderDesc);
    RenderPipelineDescriptor pipelineDesc;
    std::vector<VertexAttribute> vertexAttrib(2);
    vertexAttrib[0].shaderLocation = 0; //c
    vertexAttrib[0].format = VertexFormat::Float32x3;
    vertexAttrib[0].offset = 0;
    vertexAttrib[1].shaderLocation = 1; //r
    vertexAttrib[1].format = VertexFormat::Float32;
    vertexAttrib[1].offset = 3 * sizeof(float);
    VertexBufferLayout vertexBufferLayout;
    vertexBufferLayout.attributeCount = vertexAttrib.size();
    vertexBufferLayout.attributes = vertexAttrib.data();
    vertexBufferLayout.arrayStride = 4 * sizeof(float);
    vertexBufferLayout.stepMode = VertexStepMode::Instance;
    pipelineDesc.vertex.bufferCount = 1;
    pipelineDesc.vertex.buffers = &vertexBufferLayout;
    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;
    pipelineDesc.primitive.topology = PrimitiveTopology::TriangleList;
    pipelineDesc.primitive.stripIndexFormat = IndexFormat::Undefined;
    pipelineDesc.primitive.frontFace = FrontFace::CCW;
    pipelineDesc.primitive.cullMode = CullMode::None;
    FragmentState fragmentState;
    pipelineDesc.fragment = &fragmentState;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fs_main";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    BlendState blendState;
    blendState.color.srcFactor = BlendFactor::SrcAlpha;
    blendState.color.dstFactor = BlendFactor::OneMinusSrcAlpha;
    blendState.color.operation = BlendOperation::Add;
    blendState.alpha.srcFactor = BlendFactor::Zero;
    blendState.alpha.dstFactor = BlendFactor::One;
    blendState.alpha.operation = BlendOperation::Add;
    ColorTargetState colorTarget;
    colorTarget.format = textureDesc.format; //swapChainFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = ColorWriteMask::All;
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;
    pipelineDesc.multisample.count = 1;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;
    BindGroupLayoutEntry bindingLayout = Default;
    bindingLayout.binding = 0;
    bindingLayout.visibility = ShaderStage::Vertex | ShaderStage::Fragment;
    bindingLayout.buffer.type = BufferBindingType::Uniform;
    bindingLayout.buffer.minBindingSize = sizeof(Uniforms);
    BindGroupLayoutDescriptor bindGroupLayoutDesc;
    bindGroupLayoutDesc.entryCount = 1;
    bindGroupLayoutDesc.entries = &bindingLayout;
    BindGroupLayout bindGroupLayout = device.createBindGroupLayout(bindGroupLayoutDesc);
    PipelineLayoutDescriptor layoutDesc;
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&bindGroupLayout;
    PipelineLayout layout = device.createPipelineLayout(layoutDesc);
    pipelineDesc.layout = layout;
    RenderPipeline pipeline = device.createRenderPipeline(pipelineDesc);

    // std::vector<float> sphere_data = {
    //     -0.5, -0.5, 0., 0.2,
    //     +0.0, +0.5, 0., 0.4,
    //     +0.5, +0.5, 0., 0.2,
    //     -0.3, +0.3, 0., 0.1,
    //     +0.2, -0.4, 0., 0.1
    // };
    // Load sdf data into buffer. Scale from [0,1]^d to [-1,1]^d
    std::vector<float> sphere_data;
    sphere_data.reserve(sdf_points.rows()*4);
    for(int i=0; i<sdf_points.rows(); ++i) {
        sphere_data.push_back(sdf_points(i,0)*2. - 1.);
        sphere_data.push_back(sdf_points(i,1)*2. - 1.);
        if(dim==2) {
            sphere_data.push_back(0.);
        } else if(dim==3) {
            sphere_data.push_back(sdf_points(i,2)*2. - 1.);
        }
        sphere_data.push_back(sphere_radii(i)*2.*raster_shrink);
    }
    BufferDescriptor bufferDesc;
    bufferDesc.label = "Input buffer";
    bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Vertex;
    bufferDesc.size = sphere_data.size() * sizeof(float);
    bufferDesc.mappedAtCreation = false;
    Buffer input_buffer = device.createBuffer(bufferDesc);
    queue.writeBuffer(input_buffer, 0, sphere_data.data(), bufferDesc.size);
    bufferDesc.label = "Uniform buffer";
    bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::Uniform;
    bufferDesc.size = sizeof(Uniforms);
    bufferDesc.mappedAtCreation = false;
    Buffer uniform_buffer = device.createBuffer(bufferDesc);
    Uniforms uniforms;
    uniforms.res = res;
    uniforms.narrow_band_by = narrow_band ? 2./res : 0.;
    uniforms.z = 0.;
    uniforms.pass = 0;
    queue.writeBuffer(uniform_buffer, 0, &uniforms, sizeof(Uniforms));
    bufferDesc.label = "Output buffer";
    bufferDesc.usage = BufferUsage::CopyDst | BufferUsage::MapRead;
    uint32_t out_buffer_size = sizeof(uint32_t) * res * res;
    bufferDesc.size = out_buffer_size;
    bufferDesc.mappedAtCreation = false;
    Buffer output_buffer = device.createBuffer(bufferDesc);

    BindGroupEntry binding;
    binding.binding = 0;
    binding.buffer = uniform_buffer;
    binding.offset = 0;
    binding.size = sizeof(Uniforms);
    BindGroupDescriptor bindGroupDesc;
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = bindGroupLayoutDesc.entryCount;
    bindGroupDesc.entries = &binding;
    BindGroup bind_group = device.createBindGroup(bindGroupDesc);

    const auto render_pass = [&] (const int pass,
        const float z,
        TextureView& textureView) {
        //Update uniform buffer
        if(pass==0) {
            uniforms.z = z;
            queue.writeBuffer(uniform_buffer, offsetof(Uniforms,z),
                &uniforms.z, sizeof(float));
        }
        uniforms.pass = pass;
        queue.writeBuffer(uniform_buffer, offsetof(Uniforms,pass),
            &uniforms.pass, sizeof(int32_t));

        //Draw
        CommandEncoderDescriptor commandEncoderDesc;
        commandEncoderDesc.label = "Command Encoder";
        CommandEncoder encoder = device.createCommandEncoder(commandEncoderDesc);
        RenderPassDescriptor renderPassDesc;
        WGPURenderPassColorAttachment renderPassColorAttachment = {};
        renderPassColorAttachment.view = textureView;
        renderPassColorAttachment.resolveTarget = nullptr;
        if(pass==0) {
            renderPassColorAttachment.loadOp = LoadOp::Clear;
        } else { //pass==1
            renderPassColorAttachment.loadOp = LoadOp::Load;
        }
        renderPassColorAttachment.storeOp = StoreOp::Store;
        if(narrow_band) {
            renderPassColorAttachment.clearValue = Color{1.0, 1.0, 1.0, 1.0};
        } else {
            renderPassColorAttachment.clearValue = Color{0.0, 0.0, 0.0, 1.0};
        }
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments = &renderPassColorAttachment;
        renderPassDesc.depthStencilAttachment = nullptr;
        renderPassDesc.timestampWriteCount = 0;
        renderPassDesc.timestampWrites = nullptr;
        RenderPassEncoder renderPass = encoder.beginRenderPass(renderPassDesc);
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, input_buffer, 0,
            sphere_data.size()*sizeof(float));
        renderPass.setBindGroup(0, bind_group, 0, nullptr);
        renderPass.draw(6, sphere_data.size()/4, 0, 0);
        renderPass.end();   
        CommandBufferDescriptor cmdBufferDescriptor;
        cmdBufferDescriptor.label = "Command buffer";
        CommandBuffer command = encoder.finish(cmdBufferDescriptor);
        queue.submit(command);
    };

    std::vector<Vecd> points; //We will store rasterized points here.
    const auto read_from_texture = [&] (const float z) {
        // Copy from texture to buffer
        CommandEncoderDescriptor commandEncoderDesc;
        commandEncoderDesc.label = "Command Encoder";
        CommandEncoder encoder = device.createCommandEncoder(commandEncoderDesc);
        ImageCopyTexture source;
        source.nextInChain = nullptr;
        source.texture = texture;
        source.mipLevel = 0;
        source.origin = {0, 0, 0};
        source.aspect = TextureAspect::All;
        ImageCopyBuffer destination;
        destination.nextInChain = nullptr;
        TextureDataLayout textureDataLayout;
        textureDataLayout.nextInChain = nullptr;
        textureDataLayout.offset = 0;
        textureDataLayout.bytesPerRow = sizeof(uint32_t) * res;
        textureDataLayout.rowsPerImage = res;
        destination.layout = textureDataLayout;
        destination.buffer = output_buffer;
        encoder.copyTextureToBuffer(source, destination, textureDesc.size);
        CommandBufferDescriptor cmdBufferDescriptor;
        cmdBufferDescriptor.label = "Command buffer";
        CommandBuffer command = encoder.finish(cmdBufferDescriptor);
        queue.submit(command);

        // Copy from buffer to point array.
        struct Context {
            Buffer buffer;
            uint32_t buffer_size;
            int res;
            float z;
            std::vector<Vecd>* points;
        };
        auto on_buffer_mapped = [](WGPUBufferMapAsyncStatus status, void* cv) {
            if(status != BufferMapAsyncStatus::Success) {
                return;
            }
            Context* c = reinterpret_cast<Context*>(cv);
            uint32_t* buffer_data =
            (uint32_t*) c->buffer.getConstMappedRange(0, c->buffer_size);
            std::vector<Vecd>& points = *(c->points);

            int id = 0;
            for(int i=0; i<c->buffer_size/sizeof(uint32_t); ++i) {
                if(buffer_data[i] < 0xFFFFFFFF) {
                    //Write this point to the points array.
                    const double x = (i%c->res)/static_cast<double>(c->res),
                    y = 1. - (i/c->res)/static_cast<double>(c->res),
                    z = 0.5*(c->z+1.);
                    if constexpr(dim==2) {
                        points.emplace_back(x,y);
                    } else if constexpr(dim==3) {
                        points.emplace_back(x,y,z);
                    }
                }
                ++id;
                if(id>c->res) {
                    id = 0;
                }
            }

            c->buffer.unmap();
        };
        Context c = {output_buffer, out_buffer_size, res, z, &points};
        wgpuBufferMapAsync(output_buffer, MapMode::Read, 0, out_buffer_size,
            on_buffer_mapped, (void*)&c);
        wgpuDevicePoll(device, true, nullptr);
    };

    // Main event loop
    // while (!glfwWindowShouldClose(window)) {
    //     glfwPollEvents();
    //     TextureView textureView = swapChain.getCurrentTextureView();
    //     if (!textureView) {
    //         if(verbose) {
    //             std::cout << "Cannot acquire next swap chain texture" << std::endl;
    //         }
    //         return false;
    //     }

    if(dim==2) {
        //First render pass
        render_pass(0, 0., textureView);
        if(narrow_band) {
            //Second render pass
            render_pass(1, 0., textureView);
        }
        read_from_texture(0.);
    } else if(dim==3) {
        for(float z = -1.+0.5/res; z < 1. ; z += 1./res) {
            //First render pass
            render_pass(0, z, textureView);
            if(narrow_band) {
                //Second render pass
                render_pass(1, z, textureView);
            }
            read_from_texture(z);
        }
    }

    //     textureView.release();
    //     swapChain.present();
    // }

    // GL boilerplate shutdown
    // glfwDestroyWindow(window);
    // glfwTerminate();

    // swapChain.release();
    device.release();
    adapter.release();
    instance.release();
    // surface.release();
    input_buffer.destroy();
    input_buffer.release();
    uniform_buffer.destroy();
    uniform_buffer.release();
    output_buffer.destroy();
    output_buffer.release();
    texture.destroy();
    texture.release();

    // Finally, map points to outside array.
    outside_points.resize(points.size(), dim);
    for(int i=0; i<points.size(); ++i) {
        outside_points.row(i) = points[i];
    }

    return true;
}

#endif


template<int dim>
void outside_points_from_rasterization_cpu(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    const int res,
    const double tol,
    const bool narrow_band,
    const bool parallel,
    const bool verbose,
    Eigen::MatrixXd & outside_points)
{
    using Vecd = Eigen::Matrix<double,dim,1>;
    using Veci = Eigen::Matrix<int,dim,1>;
    using T_Grid = char;
    using GridVec = Eigen::Matrix<T_Grid, Eigen::Dynamic, 1>;

    assert(dim==sdf_points.cols());

    const double raster_shrink = 0.95;
    const bool random_perturbation = true; //randomly perturb the rasterization points.

    // Grid dimensions (have to be divisible by 2)
    const Eigen::Vector3i n(res, res, dim==2 ? 1 : res);
    // const int nx=n(0), ny=n(1), nz=n(2);
    const Eigen::Vector3d nd(n(0), n(1), n(2));
    const int n_total = n(0)*n(1)*n(2);

    // Get a bounding box of all spheres
    const Vecd bbmin_d = (sdf_points.colwise() - sphere_radii(Eigen::all,0))
    .colwise().minCoeff();
    const Vecd bbmax_d = (sdf_points.colwise() + sphere_radii(Eigen::all,0))
    .colwise().maxCoeff();
    const Veci bbmin_i = (bbmin_d.array()*nd.head<dim>().array() - 1.).floor()
        .max(0.).template cast<int>();
    const Veci bbmax_i = (bbmax_d.array()*nd.head<dim>().array() + 1.).ceil()
        .min(nd.head<dim>().array()).template cast<int>()
        .max(bbmin_i.array()+1);
    const Eigen::Vector3i bbmin(bbmin_i(0), bbmin_i(1), dim==2?0:bbmin_i(2));
    const Eigen::Vector3i bbmax(bbmax_i(0), bbmax_i(1), dim==2?1:bbmax_i(2));

    // convenience function that turns ix, iy, iz into a Veci with correct dim.
    const auto to_iter = [] (const int ix, const int iy, const int iz) {
        Veci c;
        if(dim==2) {
            c << ix, iy;
        } else {
            //dim==3
            c << ix, iy, iz;
        }
        return c;
    };

    // project a sphere coordinate into grid space
    const auto to_grid_coords = [&] (const Vecd& p) {
        Veci pr;
        for(int i=0; i<dim; ++i) {
            pr(i) = n(i)*p(i);
            if(pr(i)>=n(i)) {
                pr(i) = n(i)-1;
            }
        }
        return pr;
    };

    // Get convenient reference to a grid
    const auto grid_emb = [&] (const Veci& p) {
        if(dim==2) {
            return p(0) + n(0)*p(1);
        } else {
            //dim==3
            return p(0) + n(0)*p(1) + n(0)*n(1)*p(2);
        }
    };

    // For each sphere, get the box bounds to iterate over
    const auto box_lo = [&] (const Vecd& p, const double s) {
        const Veci pu = to_grid_coords(p);
        Eigen::Vector3i lo;
        for(int i=0; i<dim; ++i) {
            // lo(i) = std::max(0,pu(i)-static_cast<int>(s*n(i)));
            lo(i) = std::max(0,pu(i)-static_cast<int>(std::ceil(s*n(i)))-1);
        }
        if(dim==2) {
            lo(2) = 0;
        }
        return lo;
    };
    const auto box_hi = [&] (const Vecd& p, const double s) {
        const Veci pu = to_grid_coords(p);
        Eigen::Vector3i hi;
        for(int i=0; i<dim; ++i) {
            // hi(i) = std::min(n(i),pu(i)+static_cast<int>(s*n(i)));
            hi(i) = std::min(n(i),pu(i)+static_cast<int>(std::ceil(s*n(i)))+1);
        }
        if(dim==2) {
            hi(2) = 1;
        }
        return hi;
    };

    Eigen::MatrixXd grid_pts;
    if(random_perturbation) {
        grid_pts.resize(n_total,dim);
        // grid_pts.setConstant(-1.);
        std::uniform_real_distribution<double> ud(0., 1.);
        std::minstd_rand rng(rng_seed);
        for(int iz=bbmin(2); iz<bbmax(2); ++iz) {
            for(int iy=bbmin(1); iy<bbmax(1); ++iy) {
                for(int ix=bbmin(0); ix<bbmax(0); ++ix) {
                    const int i = grid_emb(to_iter(ix, iy, iz));
                    for(int j=0; j<dim; ++j) {
                        grid_pts(i,j) = ((j==0?ix:(j==1?iy:iz)) + ud(rng)) / nd(j);
                    }
                }
            }
        }
    }
    const auto get_grid_pt = [&] (const Veci& c) {
        Vecd p;
        if(random_perturbation) {
            p = grid_pts.row(grid_emb(c));
        } else {
            for(int j=0; j<dim; ++j) {
                p(j) = c(j) / nd(j);
            }
        }
        return p;
    };

    // Utility functions to check whether a grid coord is in a sphere.
    const auto in_sphere = [&] (const Veci& c, const Vecd& p, const double s) {
        const double eV = (get_grid_pt(c) - p).squaredNorm();
        return eV <= raster_shrink * std::pow(s,2);
    };

    // Rasterize a sphere with radius s and position p onto a grid.
    // Potential speedup: we don't need to perform the checks for -x AND x.
    // Enough to only perform for x. Might be premature though.
    const auto rasterize_onto_grid = [&]
    (const Vecd& p, const double s, GridVec& grid) {
        const Eigen::Vector3i lo=box_lo(p,s), hi=box_hi(p,s);
        for(int iz=lo[2]; iz<hi[2]; ++iz) {
            for(int iy=lo[1]; iy<hi[1]; ++iy) {
                for(int ix=lo[0]; ix<hi[0]; ++ix) {
                    const Veci c = to_iter(ix,iy,iz);
                    if(in_sphere(c,p,s)) {
                        if(grid(grid_emb(c)) == 0) {
                            grid(grid_emb(c)) = 1;
                        }
                        // grid(grid_emb(c)) += 1;
                    }
                }
            }
        }
    };

    //Loop through each sphere, write into grid.
    std::vector<GridVec> grids;
    const auto prep_f = [&] (const int n_threads) {
        grids.resize(n_threads);
        for(auto& grid : grids) {
            grid.resize(n_total);
            grid.setZero();
        }
    };
    const auto loop_f = [&] (const int i, const int thread_id) {
        const Vecd& p = sdf_points.row(i);
        const double s = sphere_radii(i,0);
        // This is thread-safe - I am only accessing const variables except for
        // grids, where I am only accessing my thread's grid.
        rasterize_onto_grid(p, s, grids[thread_id]);
    };
    const auto accum_f = [&] (const int thread_id) {
    };
    if(parallel) {
        igl::parallel_for(sdf_points.rows(), prep_f, loop_f, accum_f, 1000);
    } else {
        prep_f(1);
        for(int i=0; i<sdf_points.rows(); ++i) {
            loop_f(i,0);
        }
        accum_f(0);
    }
    GridVec grid;
    grid.resize(n_total);
    // grid.setZero();
    for(int iz=bbmin(2); iz<bbmax(2); ++iz) {
        for(int iy=bbmin(1); iy<bbmax(1); ++iy) {
            for(int ix=bbmin(0); ix<bbmax(0); ++ix) {
                bool occupied = false;
                const int i = grid_emb(to_iter(ix,iy,iz));
                for(const auto& g : grids) {
                    if(g(i) != 0) {
                        occupied = true;
                        break;
                    }
                }
                grid(i) = occupied ? 1 : 0;
                // T_Grid x = 0;
                // const int i = grid_emb(to_iter(ix,iy,iz));
                // for(const auto& g : grids) {
                //     x += g(i);
                // }
                // grid(i) = x;
            }
        }
    }

    if(narrow_band) {
        //Occupy any cell that is entirely surrounded by unoccupied cells
        // to reduce the number of output points.
        GridVec new_grid;
        new_grid.resize(n_total);
        // new_grid.setZero();
        for(int iz=bbmin(2); iz<bbmax(2); ++iz) {
            const int izm1 = std::max(bbmin(2),iz-1),
            izp1=std::min(bbmax(2)-1,iz+1);
            for(int iy=bbmin(1); iy<bbmax(1); ++iy) {
                const int iym1 = std::max(bbmin(1),iy-1),
                iyp1 = std::min(bbmax(1)-1,iy+1);
                for(int ix=bbmin(0); ix<bbmax(0); ++ix) {
                    const int ixm1 = std::max(bbmin(0),ix-1),
                    ixp1 = std::min(bbmax(0)-1,ix+1);
                    const int i = grid_emb(to_iter(ix,iy,iz));
                    if(grid(i) != 0) {
                        new_grid(i) = 1;
                    } else {
                        Eigen::Vector3i ixs(ixm1,ix,ixp1), iys(iym1,iy,iyp1),
                        izs(izm1,iz,izp1);
                        const bool all_unoccupied = [&]() {
                            for(const int iz : izs) {
                                for(const int iy : iys) {
                                    for(const int ix : ixs) {
                                        const int i = grid_emb(
                                            to_iter(ix,iy,iz));
                                        if(grid(i) != 0) {
                                            return false;
                                        }
                                    }
                                }
                            }
                            return true;
                        }();
                        new_grid(i) = all_unoccupied ? 1 : 0;
                    }
                }
            }
        }
        std::swap(grid, new_grid);
    }

    int n_unoccupied = 0;
    for(int iz=bbmin(2); iz<bbmax(2); ++iz) {
        for(int iy=bbmin(1); iy<bbmax(1); ++iy) {
            for(int ix=bbmin(0); ix<bbmax(0); ++ix) {
                if(grid(grid_emb(to_iter(ix,iy,iz))) == 0) {
                    ++n_unoccupied;
                }
            }
        }
    }
    outside_points.resize(n_unoccupied, dim);
    int ind=0;
    for(int iz=bbmin(2); iz<bbmax(2); ++iz) {
        for(int iy=bbmin(1); iy<bbmax(1); ++iy) {
            for(int ix=bbmin(0); ix<bbmax(0); ++ix) {
                const Veci c = to_iter(ix,iy,iz);
                if(grid(grid_emb(c)) == 0) {
                    outside_points.row(ind) = get_grid_pt(c);
                    ++ind;
                }
            }
        }
    }
}


template<int dim>
void outside_points_from_rasterization(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    const int res,
    const double tol,
    const bool narrow_band,
    const bool parallel,
    const bool force_cpu,
    const bool verbose,
    Eigen::MatrixXd & outside_points)
{
#ifdef GL_AVAILABLE
    const bool use_gpu = sdf_points.rows()>=512 && res>=64 && !force_cpu;
#else
    const bool use_gpu = false;
#endif

    if(use_gpu) {
#ifdef GL_AVAILABLE
        if(verbose) {
            std::cout << "    Rasterizing on GPU." << std::endl;
        }
        const bool gpu_success = outside_points_from_rasterization_gpu<dim>(
            sdf_points, sphere_radii,
            rng_seed, res, tol, narrow_band, parallel, verbose,
            outside_points);
        if(!gpu_success) {
            if(verbose) {
                std::cout << "    Error starting GPU, trying CPU." << std::endl;
            }
            outside_points_from_rasterization_cpu<dim>(sdf_points, sphere_radii,
            rng_seed, res, tol, narrow_band, parallel, verbose,
            outside_points);
        }
#else
        std::cout << "use_gpu has to be false if GL is not available." <<
        std::endl;
#endif
    } else {
        if(verbose) {
            std::cout << "    Rasterizing on CPU." << std::endl;
        }
        outside_points_from_rasterization_cpu<dim>(sdf_points, sphere_radii,
            rng_seed, res, tol, narrow_band, parallel, verbose,
            outside_points);
    }
}


template void outside_points_from_rasterization<2>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, int, int, double, bool, bool, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);
template void outside_points_from_rasterization<3>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, int, int, double, bool, bool, bool, bool, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);

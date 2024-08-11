#include "UploadFunctions.h"
#include "DeviceDataCoordinator.h"
#include "Device/CUDAChecks.h"
#include "Scene/Graph.h"
#include <cudaGL.h>

#include "Device/DevicePrograms/DebugDownloadData.h"
#include "Device/Wrappers/SoaWorkItems.h"
#include "Device/Wrappers/WorkItems.h"

#include "MDL/CudaLinker.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"
#include "NeuralNetworks/Interface/NetworkInterfaceUploader.h"

namespace vtx::device
{
	InstanceData createInstanceData(const std::shared_ptr<graph::Instance>& instanceNode)
	{
		//CUDA_SYNC_CHECK();

		const vtxID& instanceId = instanceNode->getUID();
		VTX_INFO("Device Visitor: Creating Instance {} data", instanceId);

		auto & geometryDataMap = onDeviceData->geometryDataMap;
		auto & materialDataMap = onDeviceData->materialDataMap;
		auto & lightDataMap    = onDeviceData->lightDataMap;

		InstanceData instanceData{};
		instanceData.instanceId = instanceNode->getTypeID();
		instanceData.transform  = instanceNode->transform->globalTransform;
		instanceNode->transform->state.updateOnDevice = false;


		const vtxID& meshId = instanceNode->getChild()->getUID();
		if (geometryDataMap.contains(meshId))
		{
			instanceData.geometryData = geometryDataMap[meshId].getDeviceImage();
		}
		else
		{
			VTX_ERROR("Requesting Mesh device Data of Mesh Node {} but not found!", meshId);
		}

		instanceData.hasEmission = false;
		instanceData.hasOpacity = false;
		std::vector<InstanceData::SlotIds> materialSlots;
		for (const graph::MaterialSlot& materialSlot : instanceNode->getMaterialSlots())
		{
			InstanceData::SlotIds slotIds{nullptr, nullptr};

			if(materialSlot.material)
			{
				if (vtxID materialId = materialSlot.material->getUID(); materialDataMap.contains(materialId))
				{
					slotIds.material = materialDataMap[materialId].getDeviceImage();

					if (vtxID lightId = materialSlot.meshLight->getUID(); lightDataMap.contains(lightId)) {
						instanceData.hasEmission = std::dynamic_pointer_cast<graph::Material>(materialSlot.material)->useEmission();
						slotIds.meshLight = lightDataMap[lightId].getDeviceImage();
					}

					if (!instanceData.hasOpacity)
					{
						const std::shared_ptr<graph::Material>& materialNode = graph::Scene::getSim()->getNode<graph::Material>(materialId);
						if (materialNode->useOpacity())
						{
							instanceData.hasOpacity = true;
						}
					}
				}
				else
				{
					VTX_ERROR("Requesting Material device Data of Material Node {} but not found!", materialId);
				}
			}

			materialSlots.push_back(slotIds);
		}
		if (!materialSlots.empty())
		{
			CUDABuffer& materialSlotsBuffer = onDeviceData->instanceDataMap.getResourceBuffers(instanceId).materialSlotsBuffer;
			materialSlotsBuffer.upload(materialSlots);
			instanceData.materialSlots = materialSlotsBuffer.castedPointer<InstanceData::SlotIds>();
			instanceData.numberOfSlots = materialSlots.size();
		}
		else
		{
			instanceData.materialSlots = nullptr;
		}

		return instanceData;
	}

	LightData createMeshLightData(const std::shared_ptr<graph::MeshLight>& meshLight)
	{
		LightData lightData;
		MeshLightAttributesData meshLightData;

		CUDABuffer& areaCdfBuffer = onDeviceData->lightDataMap.getResourceBuffers(meshLight->getUID()).areaCdfBuffer;
		areaCdfBuffer.upload(meshLight->cdfAreas);

		CUDABuffer& actualTriangleIndices = onDeviceData->lightDataMap.getResourceBuffers(meshLight->getUID()).actualTriangleIndices;
		actualTriangleIndices.upload(meshLight->actualTriangleIndices);

		vtxID meshId = meshLight->mesh->getUID();
		vtxID materialId = meshLight->material->getUID();
		GeometryData* geometryData = onDeviceData->geometryDataMap[meshId].getDeviceImage();
		MaterialData* materialData = onDeviceData->materialDataMap[materialId].getDeviceImage();

		meshLightData.instanceId                        = meshLight->parentInstanceId;
		meshLightData.geometryData                      = geometryData;
		meshLightData.materialId                        = materialData;
		meshLightData.cdfArea                           = areaCdfBuffer.castedPointer<float>();
		meshLightData.actualTriangleIndices             = actualTriangleIndices.castedPointer<uint32_t>();
		//std::shared_ptr<graph::Instance> parentInstance = graph::Scene::getSim()->getNode<graph::Instance>(meshLight->parentInstanceId);
		//std::shared_ptr<graph::Transform> transform     = parentInstance->transform;
		//math::vec3f translation, scaleVector, eulerAngles;
		//math::VectorFromAffine<math::LinearSpace3f>(transform->globalTransform, translation, scaleVector, eulerAngles);
		meshLightData.size                              = meshLight->cdfAreas.size();
		meshLightData.totalArea                         = meshLight->area;

		CUDABuffer& attributeBuffer = onDeviceData->lightDataMap.getResourceBuffers(meshLight->getUID()).attributeBuffer;
		attributeBuffer.upload(meshLightData);

		lightData.type = L_MESH;
		lightData.attributes = attributeBuffer.dPointer();

		return lightData;
	}

	LightData createEnvLightData(std::shared_ptr<graph::EnvironmentLight> envLight)
	{
		LightData lightData{};

		EnvLightAttributesData envLightData{};

		CUDABuffer& aliasBuffer = onDeviceData->lightDataMap.getResourceBuffers(envLight->getUID()).aliasBuffer;
		aliasBuffer.upload(envLight->aliasMap);

		TextureData* texture = onDeviceData->textureDataMap[envLight->envTexture->getUID()].getDeviceImage();
		envLightData.texture = texture;
		envLightData.transformation = envLight->transform->affineTransform;
		envLightData.invTransformation = envLight->transform->affineTransform.inverse();
		envLightData.aliasMap = aliasBuffer.castedPointer<AliasData>();
		envLightData.scaleLuminosity = envLight->scaleLuminosity;

		CUDABuffer& attributeBuffer = onDeviceData->lightDataMap.getResourceBuffers(envLight->getUID()).attributeBuffer;
		attributeBuffer.upload(envLightData);

		lightData.type = L_ENV;
		lightData.attributes = attributeBuffer.dPointer();
		envLight->transform->state.updateOnDevice = false;
		envLight->state.updateOnDevice = false;

		return lightData;
	}

	GeometryData createGeometryData(const std::shared_ptr<graph::Mesh>& meshNode)
	{
		VTX_INFO("Computing BLAS");

		/// Uploading Vertex and Index Buffer ///

		CUDABuffer& vertexBuffer = onDeviceData->geometryDataMap.getResourceBuffers(meshNode->getUID()).vertexBuffer;
		CUDABuffer& indexBuffer = onDeviceData->geometryDataMap.getResourceBuffers(meshNode->getUID()).indexBuffer;
		CUDABuffer& faceBuffer = onDeviceData->geometryDataMap.getResourceBuffers(meshNode->getUID()).faceBuffer;

		vertexBuffer.upload(meshNode->vertices);
		indexBuffer.upload(meshNode->indices);
		faceBuffer.upload(meshNode->faceAttributes);


		const CUdeviceptr vertexData = vertexBuffer.dPointer();
		const CUdeviceptr indexData = indexBuffer.dPointer();

		const OptixTraversableHandle traversable = optix::createGeometryAcceleration(vertexData,
			static_cast<uint32_t>(meshNode->vertices.size()),
			sizeof(graph::VertexAttributes),
			indexData,
			static_cast<uint32_t>(meshNode->indices.size()),
			sizeof(vtxID) * 3);

		GeometryData data;

		data.type = PT_TRIANGLES;
		data.traversable = traversable;
		data.vertexAttributeData = vertexBuffer.castedPointer<graph::VertexAttributes>();
		data.indicesData = indexBuffer.castedPointer<vtxID>();
		data.faceAttributeData = faceBuffer.castedPointer<graph::FaceAttributes>();
		data.numVertices = meshNode->vertices.size();
		data.numIndices = meshNode->indices.size();
		data.numFaces = meshNode->faceAttributes.size();
		return data;
	}

	DeviceShaderConfiguration* createDeviceShaderConfiguration(const std::shared_ptr<graph::Material>& material)
	{
		const graph::DevicePrograms& dp = material->getPrograms();
		const graph::Configuration& config = material->getConfiguration();
		optix::PipelineOptix* rp = optix::getRenderingPipeline();
		//const CudaMap<vtxID, optix::sbtPosition>& sbtMap = rp->getSbtMap();
		DeviceShaderConfiguration                 dvConfig;

		// The constant expression values:
		//bool thin_walled; // Stored inside flags.
		// Simplify the conditions by translating all constants unconditionally.

		dvConfig.isThinWalled = material->isThinWalled();
		dvConfig.hasOpacity = material->useOpacity();
		dvConfig.isEmissive = material->useEmission();
		dvConfig.directCallable = (getOptions()->mdlCallType == MDL_DIRECT_CALL);

		if (getOptions()->mdlCallType == MDL_DIRECT_CALL)
		{
			dvConfig.surfaceIntensity = math::vec3f(config.surfaceIntensity[0], config.surfaceIntensity[1], config.surfaceIntensity[2]);
			dvConfig.surfaceIntensityMode = config.surfaceIntensityMode;
			dvConfig.backfaceIntensity = math::vec3f(config.backfaceIntensity[0], config.backfaceIntensity[1], config.backfaceIntensity[2]);
			dvConfig.backfaceIntensityMode = config.backfaceIntensityMode;
			dvConfig.ior = math::vec3f(config.ior[0], config.ior[1], config.ior[2]);
			dvConfig.absorptionCoefficient = math::vec3f(config.absorptionCoefficient[0], config.absorptionCoefficient[1], config.absorptionCoefficient[2]);
			dvConfig.scatteringCoefficient = math::vec3f(config.scatteringCoefficient[0], config.scatteringCoefficient[1], config.scatteringCoefficient[2]);
			dvConfig.cutoutOpacity = config.cutoutOpacity;

			if (dp.pgInit) {
				dvConfig.idxCallInit = rp->getProgramSbt(dp.pgInit->name);
			}

			if (dp.pgThinWalled) {
				dvConfig.idxCallThinWalled = rp->getProgramSbt(dp.pgThinWalled->name);
			}

			if (dp.pgSurfaceScatteringSample) {
				dvConfig.idxCallSurfaceScatteringSample = rp->getProgramSbt(dp.pgSurfaceScatteringSample->name);
			}

			if (dp.pgSurfaceScatteringEval) {
				dvConfig.idxCallSurfaceScatteringEval = rp->getProgramSbt(dp.pgSurfaceScatteringEval->name);
			}

			if (dp.pgSurfaceScatteringAuxiliary) {
				dvConfig.idxCallSurfaceScatteringAuxiliary = rp->getProgramSbt(dp.pgSurfaceScatteringAuxiliary->name);
			}

			if (dp.pgBackfaceScatteringSample) {
				dvConfig.idxCallBackfaceScatteringSample = rp->getProgramSbt(dp.pgBackfaceScatteringSample->name);
			}

			if (dp.pgBackfaceScatteringEval) {
				dvConfig.idxCallBackfaceScatteringEval = rp->getProgramSbt(dp.pgBackfaceScatteringEval->name);
			}

			if (dp.pgBackfaceScatteringAuxiliary) {
				dvConfig.idxCallBackfaceScatteringAuxiliary = rp->getProgramSbt(dp.pgBackfaceScatteringAuxiliary->name);
			}

			if (dp.pgSurfaceEmissionEval) {
				dvConfig.idxCallSurfaceEmissionEval = rp->getProgramSbt(dp.pgSurfaceEmissionEval->name);
			}

			if (dp.pgSurfaceEmissionIntensity) {
				dvConfig.idxCallSurfaceEmissionIntensity = rp->getProgramSbt(dp.pgSurfaceEmissionIntensity->name);
			}

			if (dp.pgSurfaceEmissionIntensityMode) {
				dvConfig.idxCallSurfaceEmissionIntensityMode = rp->getProgramSbt(dp.pgSurfaceEmissionIntensityMode->name);
			}

			if (dp.pgBackfaceEmissionEval) {
				dvConfig.idxCallBackfaceEmissionEval = rp->getProgramSbt(dp.pgBackfaceEmissionEval->name);
			}

			if (dp.pgBackfaceEmissionIntensity) {
				dvConfig.idxCallBackfaceEmissionIntensity = rp->getProgramSbt(dp.pgBackfaceEmissionIntensity->name);
			}

			if (dp.pgBackfaceEmissionIntensityMode) {
				dvConfig.idxCallBackfaceEmissionIntensityMode = rp->getProgramSbt(dp.pgBackfaceEmissionIntensityMode->name);
			}

			if (dp.pgIor) {
				dvConfig.idxCallIor = rp->getProgramSbt(dp.pgIor->name);
			}

			if (dp.pgVolumeAbsorptionCoefficient) {
				dvConfig.idxCallVolumeAbsorptionCoefficient = rp->getProgramSbt(dp.pgVolumeAbsorptionCoefficient->name);
			}

			if (dp.pgVolumeScatteringCoefficient) {
				dvConfig.idxCallVolumeScatteringCoefficient = rp->getProgramSbt(dp.pgVolumeScatteringCoefficient->name);
			}

			if (dp.pgVolumeDirectionalBias) {
				dvConfig.idxCallVolumeDirectionalBias = rp->getProgramSbt(dp.pgVolumeDirectionalBias->name);
			}

			if (dp.pgGeometryCutoutOpacity) {
				dvConfig.idxCallGeometryCutoutOpacity = rp->getProgramSbt(dp.pgGeometryCutoutOpacity->name);
			}

			if (dp.pgHairSample) {
				dvConfig.idxCallHairSample = rp->getProgramSbt(dp.pgHairSample->name);
			}

			if (dp.pgHairEval) {
				dvConfig.idxCallHairEval = rp->getProgramSbt(dp.pgHairEval->name);
			}
		}
		else if (getOptions()->mdlCallType == MDL_INLINE)
		{

		}
		else if (getOptions()->mdlCallType == MDL_CUDA || getOptions()->mdlCallType == MDL_INLINE)
		{
			if (dp.pgEvaluateMaterial) {
				dvConfig.idxCallEvaluateMaterialStandard = rp->getProgramSbt(dp.pgEvaluateMaterial->name);
				dvConfig.idxCallEvaluateMaterialWavefront = rp->getProgramSbt(dp.pgEvaluateMaterial->name, "wfShade");
				//VTX_WARN("Fetching Shader {} EvaluateMaterial STANDARD program {} with SBT {}", material->name, dp.pgEvaluateMaterial->name, dvConfig.idxCallEvaluateMaterialStandard);
				//VTX_WARN("Fetching Shader {} EvaluateMaterial WAVEFRONT program {} with SBT {}", material->name, dp.pgEvaluateMaterial->name, dvConfig.idxCallEvaluateMaterialWavefront);
			}
			dvConfig.idxCallEvaluateMaterialWavefrontCuda = mdl::getMdlCudaLinker().getMdlFunctionIndices(material->name);
		}


		CUDABuffer& materialConfigBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).materialConfigBuffer;
		materialConfigBuffer.upload(dvConfig);
		return materialConfigBuffer.castedPointer<DeviceShaderConfiguration>();
	}

	TextureHandler* createTextureHandler(const std::shared_ptr<graph::Material>& material)
	{
		TextureHandler textureHandler{};

		auto& textureDataMap = onDeviceData->textureDataMap;
		auto& bsdfDataMap = onDeviceData->bsdfDataMap;
		auto& lightProfileDataMap = onDeviceData->lightProfileDataMap;

		// The following code was based on different assumptions on how mdl can access textures, it was probably wrong
		// It seems like mdl will provide indices to shader resource based on the order by whihch they are declared in the mdl file (target code)
		// Thi following code will try to remap the order
		if (const size_t size = material->getTextures().size(); size > 0)
		{
			std::vector<TextureData*> textures(size);
			for (const std::shared_ptr<graph::Texture> texture : material->getTextures())
			{
				vtxID          textureId = texture->getUID();
				const uint32_t mdlIndex = texture->mdlIndex;
				if (textureDataMap.contains(textureId))
				{
					textures[mdlIndex - 1] = textureDataMap[textureId].getDeviceImage();
				}
				else
				{
					VTX_ERROR("Requesting Texture device Data of texture Node {} but not found!", textureId);
				}
			}
			CUDABuffer& textureIdBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).textureIdBuffer;
			textureIdBuffer.upload(textures);
			textureHandler.numTextures = textures.size();
			textureHandler.textures = textureIdBuffer.castedPointer<TextureData*>();
		}
		else
		{
			textureHandler.numTextures = 0;
			textureHandler.textures = nullptr;
		}


		if (const size_t size = material->getBsdfs().size(); size > 0)
		{
			std::vector<BsdfData*> bsdfs(size);
			for (const std::shared_ptr<graph::BsdfMeasurement> bsdf : material->getBsdfs())
			{
				vtxID          bsdfId = bsdf->getUID();
				const uint32_t mdlIndex = bsdf->mdlIndex;
				if (bsdfDataMap.contains(bsdfId))
				{
					bsdfs[mdlIndex] = bsdfDataMap[bsdfId].getDeviceImage();
				}
				else
				{
					VTX_ERROR("Requesting Bsdf device Data of Bsdf Node {} but not found!", bsdfId);
				}
			}
			CUDABuffer& bsdfIdBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).bsdfIdBuffer;
			bsdfIdBuffer.upload(bsdfs);
			textureHandler.bsdfs = bsdfIdBuffer.castedPointer<BsdfData*>();
			textureHandler.numBsdfs = bsdfs.size();
		}
		else
		{
			textureHandler.numBsdfs = 0;
			textureHandler.bsdfs = nullptr;
		}

		if (const size_t size = material->getLightProfiles().size(); size > 0)
		{
			std::vector<LightProfileData*> lightProfiles(size);
			for (const std::shared_ptr<graph::LightProfile> lightProfile : material->getLightProfiles())
			{
				vtxID          lightProfileId = lightProfile->getUID();
				const uint32_t mdlIndex = lightProfile->mdlIndex;
				if (lightProfileDataMap.contains(lightProfileId))
				{
					lightProfiles[mdlIndex] = lightProfileDataMap[lightProfileId].getDeviceImage();
					//lightProfiles.push_back(lightProfileId);
				}
				else
				{
					VTX_ERROR("Requesting Light profile device Data of light profile Node {} but not found!", lightProfileId);
				}
			}
			CUDABuffer& lightProfileBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).lightProfileBuffer;
			lightProfileBuffer.upload(lightProfiles);
			textureHandler.lightProfiles = lightProfileBuffer.castedPointer<LightProfileData*>();
			textureHandler.numLightProfiles = lightProfiles.size();
		}
		else
		{
			textureHandler.numLightProfiles = 0;
			textureHandler.lightProfiles = nullptr;
		}



		CUDABuffer& textureHandlerBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).TextureHandlerBuffer;
		textureHandlerBuffer.upload(textureHandler);
		return textureHandlerBuffer.castedPointer<TextureHandler>();
		
	}

	MaterialData createMaterialData(const std::shared_ptr<graph::Material>& material)
	{
		MaterialData materialData; // Set everything to zero.
		CUDABuffer&  argBlockBuffer = onDeviceData->materialDataMap.getResourceBuffers(material->getUID()).argBlockBuffer;
		// If the material has an argument block, allocate and upload it.
		if (const size_t sizeArgumentBlock = material->getArgumentBlockSize(); sizeArgumentBlock > 0)
		{
			argBlockBuffer.upload(material->getArgumentBlockData(), sizeArgumentBlock);
		}

		materialData.materialConfiguration = createDeviceShaderConfiguration(material);
		materialData.textureHandler = createTextureHandler(material);

		materialData.argBlock = argBlockBuffer.castedPointer<char>();

		return materialData;
	}

	TextureData createTextureData(const std::shared_ptr<vtx::graph::Texture>& textureNode)
	{
		VTX_INFO("Creating Texture Data for Texture Node ID: {} Name: {}", textureNode->getUID(), textureNode->databaseName);
		CUDA_ARRAY3D_DESCRIPTOR descArray3D = {};
		descArray3D.Format = textureNode->format;
		descArray3D.Width = textureNode->dimension.x;
		descArray3D.Height = textureNode->dimension.y;
		descArray3D.Depth = textureNode->dimension.z;
		descArray3D.NumChannels = textureNode->dimension.w;

		const CUaddress_mode addrMode = (textureNode->shape == mi::neuraylib::ITarget_code::Texture_shape::Texture_shape_cube) ? CU_TR_ADDRESS_MODE_CLAMP : CU_TR_ADDRESS_MODE_WRAP;
		// Create filtered textureNode object
		CUDA_TEXTURE_DESC textureDesc = {};  // This contains all textureNode parameters which can be set individually or as a whole.

		// If the flag CU_TRSF_NORMALIZED_COORDINATES is not set, the only supported address mode is CU_TR_ADDRESS_MODE_CLAMP.
		textureDesc.addressMode[0] = addrMode;
		textureDesc.addressMode[1] = addrMode;
		textureDesc.addressMode[2] = addrMode;

		textureDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.

		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
		if (textureNode->effectiveGamma != 1.0f && textureNode->pixelBytesSize == 1)
		{
			textureDesc.flags |= CU_TRSF_SRGB;
		}

		textureDesc.maxAnisotropy = 1;

		// LOD 0 only by default.
		// This means when using mipmaps it's the developer's responsibility to set at least 
		// maxMipmapLevelClamp > 0.0f before calling Texture::create() to make sure mipmaps can be sampled!
		textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDesc.mipmapLevelBias = 0.0f;
		textureDesc.minMipmapLevelClamp = 0.0f;
		textureDesc.maxMipmapLevelClamp = 0.0f; // This should be set to Picture::getNumberOfLevels() when using mipmaps.

		textureDesc.borderColor[0] = 0.0f;
		textureDesc.borderColor[1] = 0.0f;
		textureDesc.borderColor[2] = 0.0f;
		textureDesc.borderColor[3] = 0.0f;

		// Create unfiltered textureNode object if necessary (cube textures have no texel functions)
		CUDA_TEXTURE_DESC textureDescUnfiltered = textureDesc;

		if (textureNode->shape != mi::neuraylib::ITarget_code::Texture_shape_cube)
		{
			// Use a black border for access outside of the textureNode
			textureDescUnfiltered.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
			textureDescUnfiltered.filterMode = CU_TR_FILTER_MODE_POINT;
		}

		TextureData textureData;

		CUresult             result;
		CUarray&             textureArray     = onDeviceData->textureDataMap.getResourceBuffers(textureNode->getUID()).textureArray;
		cudaTextureObject_t& texObj           = onDeviceData->textureDataMap.getResourceBuffers(textureNode->getUID()).texObj;
		cudaTextureObject_t& texObjUnfiltered = onDeviceData->textureDataMap.getResourceBuffers(textureNode->getUID()).texObjUnfiltered;

		const CUDA_RESOURCE_DESC resourceDescription = uploadTexture(textureNode->imageLayersPointers, descArray3D, textureNode->pixelBytesSize, textureArray);


		result = cuTexObjectCreate(&texObj, &resourceDescription, &textureDesc, nullptr);
		CU_CHECK(result);
		result = cuTexObjectCreate(&texObjUnfiltered, &resourceDescription, &textureDescUnfiltered, nullptr);
		CU_CHECK(result);

		textureData.texObj = texObj;
		textureData.texObjUnfiltered = texObjUnfiltered;
		textureData.dimension = textureNode->dimension;
		textureData.invSize = math::vec3f{ 1.0f / textureData.dimension.x, 1.0f / textureData.dimension.y, 1.0f / textureData.dimension.z };

		return textureData;
	}

	BsdfSamplingPartData createBsdfPartData(const graph::BsdfMeasurement::BsdfPartData& bsdfData, BsdfPartBuffer& buffers)
	{
		BsdfSamplingPartData bsdfSamplingData;
		CUDABuffer& sampleData = buffers.sampleData;
		CUDABuffer& albedoData = buffers.albedoData;
		CUtexObject& evalData = buffers.evalData;

		sampleData.upload(bsdfData.sampleData);
		albedoData.upload(bsdfData.albedoData);

		bsdfSamplingData.sampleData = sampleData.castedPointer<float>();
		bsdfSamplingData.albedoData = albedoData.castedPointer<float>();
		bsdfSamplingData.maxAlbedo = bsdfData.maxAlbedo;
		bsdfSamplingData.angularResolution = bsdfData.angularResolution;
		bsdfSamplingData.numChannels = bsdfData.numChannels;
		bsdfSamplingData.invAngularResolution = math::vec2f(1.0f / static_cast<float>(bsdfData.angularResolution.x), 1.0f / static_cast<float>(bsdfData.angularResolution.y));

		// Prepare 
		// uation data:
		// - Simply store the measured data in a volume texture.
		// - In case of color data, we store each sample in a vector4 to get texture support.

		// Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
		CUDA_ARRAY3D_DESCRIPTOR descArray3D;

		descArray3D.Width = bsdfData.angularResolution.y;
		descArray3D.Height = bsdfData.angularResolution.x;
		descArray3D.Depth = bsdfData.angularResolution.x;
		descArray3D.Format = CU_AD_FORMAT_FLOAT;
		descArray3D.NumChannels = (bsdfData.numChannels == 3) ? 4 : 1;
		descArray3D.Flags = 0;

		std::vector<const void*> pointers;
		pointers.push_back(bsdfData.lookupData.data());

		const CUDA_RESOURCE_DESC resourceDescription = uploadTexture(pointers, descArray3D, sizeof(float), buffers.lookUpArray);

		bsdfSamplingData.deviceMbsdfData = buffers.lookUpArray;

		CUDA_TEXTURE_DESC textureDescription = {};  // This contains all texture parameters which can be set individually or as a whole.
		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDescription.flags = CU_TRSF_NORMALIZED_COORDINATES;
		textureDescription.filterMode = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
		textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.maxAnisotropy = 1;
		// DAR The default initialization handled all these. Just for code clarity.
		textureDescription.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDescription.mipmapLevelBias = 0.0f;
		textureDescription.minMipmapLevelClamp = 0.0f;
		textureDescription.maxMipmapLevelClamp = 0.0f;
		textureDescription.borderColor[0] = 0.0f;
		textureDescription.borderColor[1] = 0.0f;
		textureDescription.borderColor[2] = 0.0f;
		textureDescription.borderColor[3] = 0.0f;

		const CUresult result = cuTexObjectCreate(&evalData, &resourceDescription, &textureDescription, nullptr);
		CU_CHECK(result);

		bsdfSamplingData.evalData = evalData;
		return bsdfSamplingData;
	}

	BsdfData createBsdfData(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement)
	{
		BsdfData bsdfData;
		if (bsdfMeasurement->reflectionBsdf.isValid)
		{
			BsdfPartBuffer&   reflectionPartBuffer     = onDeviceData->bsdfDataMap.getResourceBuffers(bsdfMeasurement->getUID()).reflectionPartBuffer;
			const BsdfSamplingPartData reflectionBsdfDeviceData = createBsdfPartData(bsdfMeasurement->reflectionBsdf, reflectionPartBuffer);

			CUDABuffer& partBuffer = reflectionPartBuffer.partBuffer;
			partBuffer.upload(reflectionBsdfDeviceData);
			bsdfData.reflectionBsdf = partBuffer.castedPointer<BsdfSamplingPartData>();
			bsdfData.hasReflectionBsdf = true;

		}
		if (bsdfMeasurement->transmissionBsdf.isValid)
		{
			BsdfPartBuffer&   transmissionPartBuffer     = onDeviceData->bsdfDataMap.getResourceBuffers(bsdfMeasurement->getUID()).transmissionPartBuffer;
			const BsdfSamplingPartData transmissionBsdfDeviceData = createBsdfPartData(bsdfMeasurement->transmissionBsdf, transmissionPartBuffer);

			CUDABuffer& partBuffer = transmissionPartBuffer.partBuffer;
			partBuffer.upload(transmissionBsdfDeviceData);
			bsdfData.transmissionBsdf = partBuffer.castedPointer<BsdfSamplingPartData>();
			bsdfData.hasTransmissionBsdf = true;
		}
		return bsdfData;
	}

	LightProfileData createLightProfileData(std::shared_ptr<graph::LightProfile> lightProfile)
	{
		// Copy entire CDF data buffer to GPU
		CUDABuffer& cdfBuffer = onDeviceData->lightProfileDataMap.getResourceBuffers(lightProfile->getUID()).cdfBuffer;
		cdfBuffer.upload(lightProfile->lightProfileData.cdfData);

		// --------------------------------------------------------------------------------------------
		// Prepare evaluation data.
		//  - Use a 2d texture that allows bilinear interpolation.
		// Allocate a 3D array on the GPU (phi_delta x theta_out x theta_in)
		CUDA_ARRAY3D_DESCRIPTOR descArray3D = {};

		descArray3D.Width       = lightProfile->lightProfileData.resolution.x;
		descArray3D.Height      = lightProfile->lightProfileData.resolution.y;
		descArray3D.Depth       = 0; // A 2D array is allocated if only Depth extent is zero.
		descArray3D.Format      = CU_AD_FORMAT_FLOAT;
		descArray3D.NumChannels = 1;
		descArray3D.Flags       = 0;
		std::vector<const void*> pointers;
		pointers.push_back(lightProfile->lightProfileData.sourceData);

		CUarray&                 lightProfileSourceArray = onDeviceData->lightProfileDataMap.getResourceBuffers(lightProfile->getUID()).lightProfileSourceArray;
		const CUDA_RESOURCE_DESC resourceDescription     = uploadTexture(pointers, descArray3D, sizeof(float), lightProfileSourceArray);


		CUDA_TEXTURE_DESC textureDescription = {}; // This contains all texture parameters which can be set individually or as a whole.
		// Possible flags: CU_TRSF_READ_AS_INTEGER, CU_TRSF_NORMALIZED_COORDINATES, CU_TRSF_SRGB
		textureDescription.flags          = CU_TRSF_NORMALIZED_COORDINATES;
		textureDescription.filterMode     = CU_TR_FILTER_MODE_LINEAR; // Bilinear filtering by default.
		textureDescription.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP; // FIXME Shouldn't phi use wrap?
		textureDescription.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
		textureDescription.maxAnisotropy  = 1;
		// DAR The default initialization handled all these. Just for code clarity.
		textureDescription.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
		textureDescription.mipmapLevelBias = 0.0f;
		textureDescription.minMipmapLevelClamp = 0.0f;
		textureDescription.maxMipmapLevelClamp = 0.0f;
		textureDescription.borderColor[0] = 1.0f; // DAR DEBUG Why 1.0f? Shouldn't matter with clamp.
		textureDescription.borderColor[1] = 1.0f;
		textureDescription.borderColor[2] = 1.0f;
		textureDescription.borderColor[3] = 1.0f;
		CUtexObject& evalData = onDeviceData->lightProfileDataMap.getResourceBuffers(lightProfile->getUID()).evalData;
		CU_CHECK(cuTexObjectCreate(&evalData, &resourceDescription, &textureDescription, nullptr));

		LightProfileData lightProfileData{};
		lightProfileData.lightProfileArray = lightProfileSourceArray;
		lightProfileData.evalData = evalData;
		lightProfileData.cdfData = cdfBuffer.castedPointer<float>();
		lightProfileData.angularResolution = lightProfile->lightProfileData.resolution;
		lightProfileData.invAngularResolution = math::vec2f(1.0f / static_cast<float>(lightProfileData.angularResolution.x), 1.0f / static_cast<float>(lightProfileData.angularResolution.y));
		lightProfileData.thetaPhiStart = lightProfile->lightProfileData.start;        // start of the grid
		lightProfileData.thetaPhiDelta = lightProfile->lightProfileData.delta;        // angular step size
		lightProfileData.thetaPhiInvDelta.x = (lightProfileData.thetaPhiDelta.x != 0.0f) ? 1.0f / lightProfileData.thetaPhiDelta.x : 0.0f;
		lightProfileData.thetaPhiInvDelta.y = (lightProfileData.thetaPhiDelta.y != 0.0f) ? 1.0f / lightProfileData.thetaPhiDelta.y : 0.0f;
		lightProfileData.candelaMultiplier = static_cast<float>(lightProfile->lightProfileData.candelaMultiplier);
		lightProfileData.totalPower = static_cast<float>(lightProfile->lightProfileData.totalPower * lightProfile->lightProfileData.candelaMultiplier);

		return lightProfileData;
	}

	QueuesData createQueuesData(const int width, const int height, const int maxBounces)
	{
		int totPixels = width * height;
		const int maxShadowQueue = totPixels * (maxBounces + 1);
		const int maxAccumulationQueue = totPixels * (maxBounces + 1); // each bounce can produce an emissive hit

		bool create = false;
		DeviceDataCoordinator::PrevWorkQueueInfo prevQueuesInfo = onDeviceData->prevWorkQueueInfo;
		if(!prevQueuesInfo.wasAllocated)
		{
			create = true;
		}
		else
		{
			if(prevQueuesInfo.numberOfPixels != totPixels || prevQueuesInfo.maxBounces != maxBounces)
			{
				create = true;
			}
		}

		QueuesData queuesData;
		if(create)
		{
			VTX_INFO("Creating Wavefront Queues with Size: Width: {}, Height: {}, MaxBounces: {}, TotPixels: {}, MaxShadowQueue: {}, MaxAccumulationQueue: {}", width, height, maxBounces, totPixels, maxShadowQueue, maxAccumulationQueue);
			CUDABufferManager::deallocateAll();
			const WorkQueueSOA<TraceWorkItem> radianceTraceQueue(totPixels, "radianceTraceQueue");
			queuesData.radianceTraceQueue = onDeviceData->workQueuesData.resourceBuffers.radianceTraceQueueBuffer.upload(radianceTraceQueue);

			const WorkQueueSOA<RayWorkItem> shadeQueue(totPixels, "shadeQueue");
			queuesData.shadeQueue = onDeviceData->workQueuesData.resourceBuffers.shadeQueueBuffer.upload(shadeQueue);

			const WorkQueueSOA<EscapedWorkItem> escapedQueue(totPixels, "escapedQueue");
			queuesData.escapedQueue = onDeviceData->workQueuesData.resourceBuffers.escapedQueueBuffer.upload(escapedQueue);

			const WorkQueueSOA<AccumulationWorkItem> accumulationQueue(maxShadowQueue, "accumulationQueue");
			queuesData.accumulationQueue = onDeviceData->workQueuesData.resourceBuffers.accumulationQueueBuffer.upload(accumulationQueue);

			const WorkQueueSOA<ShadowWorkItem> shadowQueue(maxAccumulationQueue, "shadowQueue");
			queuesData.shadowQueue = onDeviceData->workQueuesData.resourceBuffers.shadowQueueBuffer.upload(shadowQueue);

			Counters counters{};
			queuesData.queueCounters = onDeviceData->workQueuesData.resourceBuffers.countersBuffer.upload(counters);

			onDeviceData->prevWorkQueueInfo.numberOfPixels = totPixels;
			onDeviceData->prevWorkQueueInfo.maxBounces = maxBounces;
			onDeviceData->prevWorkQueueInfo.wasAllocated = true;
		}

		return queuesData;
	}

	FrameBufferData prepareFrameBuffers(const int width, const int height)
	{
 		FrameBufferData frameBufferData;
		frameBufferData.outputBuffer = (CUdeviceptr)(onDeviceData->frameBufferData.resourceBuffers.cudaOutputBuffer.alloc<math::vec4f>(width * height));
		frameBufferData.radianceAccumulator = onDeviceData->frameBufferData.resourceBuffers.rawRadiance.alloc<math::vec3f>(width * height);
		frameBufferData.albedoAccumulator = onDeviceData->frameBufferData.resourceBuffers.albedo.alloc<math::vec3f>(width * height);
		frameBufferData.normalAccumulator = onDeviceData->frameBufferData.resourceBuffers.normal.alloc<math::vec3f>(width * height);
		frameBufferData.tmRadiance = onDeviceData->frameBufferData.resourceBuffers.tmRadiance.alloc<math::vec3f>(width * height);
		frameBufferData.hdriRadiance = onDeviceData->frameBufferData.resourceBuffers.hdriRadiance.alloc<math::vec3f>(width * height);
		frameBufferData.normalNormalized = onDeviceData->frameBufferData.resourceBuffers.normalNormalized.alloc<math::vec3f>(width * height);
		frameBufferData.albedoNormalized = onDeviceData->frameBufferData.resourceBuffers.albedoNormalized.alloc<math::vec3f>(width * height);
		frameBufferData.trueNormal = onDeviceData->frameBufferData.resourceBuffers.trueNormal.alloc<math::vec3f>(width * height);
		frameBufferData.tangent = onDeviceData->frameBufferData.resourceBuffers.tangent.alloc<math::vec3f>(width * height);
		frameBufferData.orientation = onDeviceData->frameBufferData.resourceBuffers.orientation.alloc<math::vec3f>(width * height);
		frameBufferData.uv = onDeviceData->frameBufferData.resourceBuffers.uv.alloc<math::vec3f>(width * height);
		frameBufferData.debugColor1 = onDeviceData->frameBufferData.resourceBuffers.debugColor1.alloc<math::vec3f>(width * height);
		frameBufferData.fireflyPass = onDeviceData->frameBufferData.resourceBuffers.fireflyRemoval.alloc<math::vec3f>(width * height);
		frameBufferData.samples = onDeviceData->frameBufferData.resourceBuffers.samples.alloc<int>(width * height);
		frameBufferData.gBufferHistory = onDeviceData->frameBufferData.resourceBuffers.gBufferData.alloc<gBufferHistory>(width * height);
		frameBufferData.gBuffer = onDeviceData->frameBufferData.resourceBuffers.gBuffer.alloc<float>(width * height);
		frameBufferData.noiseBuffer = onDeviceData->frameBufferData.resourceBuffers.noiseDataBuffer.alloc<NoiseData>(width * height);

		frameBufferData.frameSize.x = width;
		frameBufferData.frameSize.y = height;

		return frameBufferData;
	}

	void prepareNoiseComputationBuffers(const std::shared_ptr<graph::Renderer>& rendererNode)
	{
		const int size = rendererNode->width * rendererNode->height;
		constexpr int threadsPerBlock = 256;
		const int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

		onDeviceData->noiseComputationData.resourceBuffers.radianceRangeBuffer.resize(numBlocks * sizeof(math::vec2f));
		onDeviceData->noiseComputationData.resourceBuffers.albedoRangeBuffer.resize(numBlocks * sizeof(math::vec2f));
		onDeviceData->noiseComputationData.resourceBuffers.normalRangeBuffer.resize(numBlocks * sizeof(math::vec2f));
		onDeviceData->noiseComputationData.resourceBuffers.globalRadianceRangeBuffer.resize(sizeof(math::vec2f));
		onDeviceData->noiseComputationData.resourceBuffers.globalAlbedoRangeBuffer.resize(sizeof(math::vec2f));
		onDeviceData->noiseComputationData.resourceBuffers.globalNormalRangeBuffer.resize(sizeof(math::vec2f));
	}

	void setRendererData(const std::shared_ptr<graph::Renderer>& rendererNode)
	{
		if (rendererNode->resized) {

			//rendererNode->threadData.bufferUpdateReady = false;

			onDeviceData->launchParamsData.editableHostImage().frameBuffer = prepareFrameBuffers(rendererNode->width, rendererNode->height);

			optix::getState()->denoiser.resize(rendererNode->width, rendererNode->height);

			//TODO Smarter way to set these value, I don't want to waste memory if we are not using adaptive samplig
			{
				prepareNoiseComputationBuffers(rendererNode);
			}

			rendererNode->resized             = false;
			rendererNode->resizeGlBuffer      = true;
		}


		//Wavefront Architecture
		{
			try {
				if (rendererNode->waveFrontIntegrator.settings.active)
				{
					QueuesData queuesData = createQueuesData(rendererNode->width, rendererNode->height, rendererNode->settings.maxBounces);
					if (queuesData.queueCounters != nullptr)
					{
						onDeviceData->launchParamsData.editableHostImage().queues = queuesData;
					}
				}
				else if (onDeviceData->launchParamsData.getHostImage().queues.queueCounters != nullptr)
				{
					// Free memory if we are not using wavefront architecture
					CUDABufferManager::deallocateAll();
					onDeviceData->launchParamsData.editableHostImage().queues = QueuesData();
					onDeviceData->prevWorkQueueInfo = DeviceDataCoordinator::PrevWorkQueueInfo();
				}
			}
			catch (const std::exception& e) {
				VTX_ERROR("Error while allocating memory for wavefront architecture");
				VTX_ERROR("Error: {}", e.what());
				rendererNode->waveFrontIntegrator.settings.active = false;
				rendererNode->waveFrontIntegrator.settings.isUpdated = true;
			}
			
			if (rendererNode->waveFrontIntegrator.settings.isUpdated)
			{
				onDeviceData->launchParamsData.editableHostImage().settings.wavefront = rendererNode->waveFrontIntegrator.settings;
				rendererNode->waveFrontIntegrator.settings.isUpdated = false;
			}
		}

		{
			try
			{
				if (rendererNode->waveFrontIntegrator.network.settings.active && rendererNode->waveFrontIntegrator.settings.active)
				{
					const int totPixels = rendererNode->width * rendererNode->height;
					const int maxDatasetSize = totPixels * rendererNode->settings.maxBounces*2;// rendererNode->waveFrontIntegrator.network.settings.batchSize;
					const int maxBounces = rendererNode->settings.maxBounces;
					const network::config::DistributionType distributionType = rendererNode->waveFrontIntegrator.network.settings.distributionType;
					const int mixtureSize = rendererNode->waveFrontIntegrator.network.settings.mixtureSize;

					if (needNetworkInterfaceReallocation(totPixels,maxBounces,maxDatasetSize,distributionType,mixtureSize))
					{
						VTX_INFO("Creating network interface");
						onDeviceData->launchParamsData.editableHostImage().networkInterface
							= uploadNetworkInterface(
								totPixels,
								maxBounces,
								maxDatasetSize,
								distributionType,
								mixtureSize);
					}
				}
				else
				{
					resetNetworkInterfaceAllocation();
				}
			}
			catch (const std::exception& e) {
				VTX_ERROR("Error while allocating memory for Neural Network Training Data");
				VTX_ERROR("Error: {}", e.what());
				rendererNode->waveFrontIntegrator.network.settings.active = false;
				rendererNode->waveFrontIntegrator.network.settings.isUpdated = true;
			}
			if (rendererNode->waveFrontIntegrator.network.settings.isAnyUpdated())
			{
				onDeviceData->launchParamsData.editableHostImage().settings.neural = rendererNode->waveFrontIntegrator.network.settings;
				rendererNode->waveFrontIntegrator.network.settings.resetUpdate();
			}
		}

		DebugData::prepare(
			rendererNode->settings.maxBounces,
			rendererNode->settings.debugPixel,
			rendererNode->settings.debugDepth,
			rendererNode->waveFrontIntegrator.network.settings.mixtureSize,
			rendererNode->waveFrontIntegrator.network.settings.distributionType
		);

		if (rendererNode->settings.isAnyUpdated())
		{
			onDeviceData->launchParamsData.editableHostImage().settings.renderer = rendererNode->settings;
			rendererNode->settings.resetUpdate();
		}
	}

	CameraData createCameraData(const std::shared_ptr<graph::Camera>& cameraNode)
	{
		CameraData cameraData;
		cameraData.position = cameraNode->position;
		cameraData.direction = cameraNode->direction;
		const float radianFov = (cameraNode->fovY * M_PI) / 180.0f;
		cameraData.vertical = tanf(radianFov * 0.5f) * cameraNode->vertical;
		cameraData.horizontal = tanf(radianFov * 0.5f) * cameraNode->aspect * cameraNode->horizontal;

		return cameraData;

	}

	CUDA_RESOURCE_DESC uploadTexture(
		const std::vector<const void*>& imageLayers,
		const CUDA_ARRAY3D_DESCRIPTOR&  descArray3D,
		const size_t&                   sizeBytesPerElement,
		CUarray&                        array)
	{
		CUresult result;


		result = cuArray3DCreate(&array, &descArray3D);
		CU_CHECK(result);

		for (size_t i = 0; i < imageLayers.size(); ++i)
		{
			const void* layerSource = imageLayers[i];

			CUDA_MEMCPY3D params = {};

			params.srcMemoryType = CU_MEMORYTYPE_HOST;
			params.srcHost       = layerSource;
			params.srcPitch      = descArray3D.Width * sizeBytesPerElement * descArray3D.NumChannels;
			params.srcHeight     = descArray3D.Height;

			params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
			params.dstArray      = array;

			if (descArray3D.Depth != 0)
			{
				params.dstXInBytes = 0;
				params.dstY        = 0;
				params.dstZ        = i;
			}

			params.WidthInBytes = params.srcPitch;
			params.Height       = descArray3D.Height;
			params.Depth        = 1;

			result = cuMemcpy3D(&params);
			CU_CHECK(result);
		}

		CUDA_RESOURCE_DESC resourceDescription = {};
		resourceDescription.resType            = CU_RESOURCE_TYPE_ARRAY;
		resourceDescription.res.array.hArray   = array;

		return resourceDescription;
	}
}

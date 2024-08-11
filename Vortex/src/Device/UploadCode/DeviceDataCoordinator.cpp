#include "DeviceDataCoordinator.h"

#include "UploadFunctions.h"
#include "Device/OptixWrapper.h"
#include "Device/DevicePrograms/DebugDownloadData.h"
#include "Scene/Graph.h"
#include "Scene/Scene.h"

namespace vtx::device
{
	DeviceDataCoordinator* DeviceDataCoordinator::getInstance()
	{
		static DeviceDataCoordinator uploadDataInstance;
		return &uploadDataInstance;
	}

	void DeviceDataCoordinator::sync()
	{
		cleanDeletedNodes();

		std::shared_ptr<graph::SceneIndexManager> sim = graph::Scene::getSim();
		const std::vector<std::shared_ptr<graph::Mesh>>             meshes            = sim->getAllNodeOfType<graph::Mesh>(graph::NT_MESH);
		for (const std::shared_ptr<graph::Mesh>& mesh : meshes)
		{
			syncNode(mesh);
		}
		const std::vector<std::shared_ptr<graph::Texture>>          textures = sim->getAllNodeOfType<graph::Texture>(graph::NT_MDL_TEXTURE);
		for (const std::shared_ptr<graph::Texture>& texture : textures)
		{
			syncNode(texture);
		}
		const std::vector<std::shared_ptr<graph::BsdfMeasurement>>  bsdfMeasurements = sim->getAllNodeOfType<graph::BsdfMeasurement>(graph::NT_MDL_BSDF);
		for (const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurement : bsdfMeasurements)
		{
			syncNode(bsdfMeasurement);
		}
		const std::vector<std::shared_ptr<graph::LightProfile>>     lightProfiles = sim->getAllNodeOfType<graph::LightProfile>(graph::NT_MDL_LIGHTPROFILE);
		for (const std::shared_ptr<graph::LightProfile>& lightProfile : lightProfiles)
		{
			syncNode(lightProfile);
		}
		const std::vector<std::shared_ptr<graph::Material>>         materials = sim->getAllNodeOfType<graph::Material>(graph::NT_MATERIAL);
		for (const std::shared_ptr<graph::Material>& material : materials)
		{
			syncNode(material);
		}
		const std::vector<std::shared_ptr<graph::MeshLight>>        meshLights = sim->getAllNodeOfType<graph::MeshLight>(graph::NT_MESH_LIGHT);
		for (const std::shared_ptr<graph::MeshLight>& meshLight : meshLights)
		{
			syncNode(meshLight);
		}
		const std::vector<std::shared_ptr<graph::Instance>>         instances = sim->getAllNodeOfType<graph::Instance>(graph::NT_INSTANCE);
		for (const std::shared_ptr<graph::Instance>& instance : instances)
		{
			syncNode(instance);
		}
		const std::vector<std::shared_ptr<graph::EnvironmentLight>> environmentLights = sim->getAllNodeOfType<graph::EnvironmentLight>(graph::NT_ENV_LIGHT);
		for (const std::shared_ptr<graph::EnvironmentLight>& environmentLight : environmentLights)
		{
			syncNode(environmentLight);
		}
		const std::shared_ptr<graph::Renderer> renderer = graph::Scene::get()->renderer;
		const std::shared_ptr<graph::Camera>   camera   = renderer->camera;
		syncNode(camera);
		syncNode(renderer);

		finalize();
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Instance>& instance)
	{
		// If the child node is a mesh, then it's leaf therefore we can safely create the instance.
		// This supposes that child and transform are traversed before the instance visitor is accepted.
		if (const std::shared_ptr<graph::Mesh> meshNode = std::dynamic_pointer_cast<graph::Mesh>(instance->getChild())) {
			// TODO Check if meshes or material have been changed
			if (const vtxID instanceId = instance->getUID();
				!instanceDataMap.contains(instance->getTypeID())
				|| instance->transform->state.updateOnDevice
				|| instance->state.updateOnDevice
				)
			{
				const vtxID instanceTypeId = instance->getTypeID();
				const vtxID meshId = meshNode->getUID();

				const OptixTraversableHandle& traversable = geometryDataMap[meshId].getHostImage().traversable;
				const math::affine3f& globalTransform = instance->transform->globalTransform;
				const OptixInstance   optixInstance = optix::createInstance(instanceTypeId, globalTransform, traversable);
				const InstanceData    instanceData = createInstanceData(instance);

				instanceDataMap.insert(instanceTypeId, instanceData);
				optixInstances[instanceTypeId] = (optixInstance);

				if (!instanceDataMap.contains(instanceTypeId))
				{
					// new instance, TLAS needs rebuild;
					launchParamsData.editableHostImage().topObject = 0;
				}
				instance->state.updateOnDevice = false;
				ops::restartRender();
			}
		}
		//popTransform();
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Transform>& transform)
	{
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Group>& group)
	{
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Mesh>& mesh)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID meshId = mesh->getUID();
			!geometryDataMap.contains(meshId)
			|| mesh->state.updateOnDevice
			) {
			const GeometryData geo = createGeometryData(mesh);
			geometryDataMap.insert(meshId, geo);

			//TODO: Temporary solution since geometry is not editable
			// when we add geometry editing, the BLAS can be updated if the number of vertices indices stay the same
			// hence the TLAS can be updated too.
			// For now, geometry can only be added or removed, so we need to rebuild the TLAS
			// We mark the top Traversable as zero to force the TLAS rebuild
			launchParamsData.editableHostImage().topObject = 0;
			mesh->state.updateOnDevice = false;
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Material>& material)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID materialId = material->getUID();
			!materialDataMap.contains(materialId) ||
			material->state.updateOnDevice ||
			material->materialGraph->state.isShaderArgBlockUpdated
			) {
			const MaterialData  mat = createMaterialData(material);
			materialDataMap.insert(materialId, mat);
			material->state.updateOnDevice = false;
			material->materialGraph->resetIsShaderArgBlockUpdated();
			ops::restartRender();
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Camera>& camera)
	{
		if (camera->state.updateOnDevice) {
			launchParamsData.editableHostImage().cameraData = createCameraData(camera);
			camera->state.updateOnDevice = false;
			ops::restartRender();
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Renderer>& renderer)
	{
		setRendererData(renderer);
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::Texture>& textureNode)
	{
		//TODO : Check if the texture has been updated
		if (const vtxID textureId = textureNode->getUID(); !textureDataMap.contains(textureId)) {
			const TextureData textureData = createTextureData(textureNode);
			textureDataMap.insert(textureNode->getUID(), textureData);
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurementNode)
	{
		///TODO : Check if the texture has been updated
		if (const vtxID bsdfId = bsdfMeasurementNode->getUID(); !bsdfDataMap.contains(bsdfId)) {
			const BsdfData bsdfData = createBsdfData(bsdfMeasurementNode);
			bsdfDataMap.insert(bsdfId, bsdfData);
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::LightProfile>& lightProfile)
	{
		///TODO : Check if the light Profile has been updated
		if (const vtxID lightProfileId = lightProfile->getUID(); !lightProfileDataMap.contains(lightProfileId)) {
			const LightProfileData lightProfileData = createLightProfileData(lightProfile);
			lightProfileDataMap.insert(lightProfileId, lightProfileData);
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::MeshLight>& meshLight)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID lightId = meshLight->getUID(); !lightDataMap.contains(lightId)) {
			LightData lightData = createMeshLightData(meshLight);
			// We keep a list on the device of all potential mesh lights but only the mesh lights associated with a material
			// markes as light will be sampled in the mis.
			if (meshLight->material->useEmission())
			{
				lightData.use = true;
				lightDataMap.insert(lightId, lightData);
			}
			else
			{
				lightData.use = false;
				lightDataMap.insert(lightId, lightData);
			}
		}
	}

	void DeviceDataCoordinator::syncNode(const std::shared_ptr<graph::EnvironmentLight>& envLight)
	{
		//TODO : Check if the mesh has been updated
		if (const vtxID lightId = envLight->getUID(); !lightDataMap.contains(lightId) || envLight->transform->state.updateOnDevice || envLight->state.updateOnDevice
			) {
			LightData lightData = createEnvLightData(envLight);
			// TODO : How do we handle the presence of another env Light?
			lightData.use = true;
			lightDataMap.insert(lightId, lightData);
			launchParamsData.editableHostImage().envLight = lightDataMap[lightId].getDeviceImage();
			ops::restartRender();
		}
	}
	void DeviceDataCoordinator::cleanDeletedNodes()
	{
		cleanDeletedNodeOfType(geometryDataMap, graph::NT_MESH);
		cleanDeletedNodeOfType(materialDataMap, graph::NT_MATERIAL);
		cleanDeletedNodeOfType(textureDataMap, graph::NT_MDL_TEXTURE);
		cleanDeletedNodeOfType(bsdfDataMap, graph::NT_MDL_BSDF);
		cleanDeletedNodeOfType(lightProfileDataMap, graph::NT_MDL_LIGHTPROFILE);
		cleanDeletedNodeOfType(lightDataMap, graph::NT_LIGHT);
		cleanDeletedNodeOfType(lightDataMap, graph::NT_MESH_LIGHT);
		cleanDeletedNodeOfType(lightDataMap, graph::NT_ENV_LIGHT);
		cleanDeletedNodeOfType(instanceDataMap, graph::NT_INSTANCE);
	}
	void DeviceDataCoordinator::finalize()
	{
		if (instanceDataMap.isMapChanged && instanceDataMap.size() != 0)
		{
			std::vector<InstanceData*> instances;
			instanceDataMap.isMapChanged = false;

			for (const vtxID instanceId : instanceDataMap)
			{
				InstanceData* instanceData = instanceDataMap[instanceId].getDeviceImage();
				if (instanceId >= instances.size())
				{
					instances.resize(instanceId + 1);
				}
				instances[instanceId] = instanceData;
			}

			std::vector<OptixInstance> optixInstancesVector;
			optixInstancesVector.reserve(optixInstances.size());
			for (const std::pair<const unsigned, OptixInstance>& instance : optixInstances)
			{
				optixInstancesVector.push_back(instance.second);
			}

			OptixAabb aabb;
			launchParamsData.editableHostImage().topObject = optix::createInstanceAcceleration(optixInstancesVector, launchParamsData.editableHostImage().topObject, &aabb);
			launchParamsData.editableHostImage().aabb = aabb;
			launchParamsData.editableHostImage().instances = launchParamsData.resourceBuffers.instancesBuffer.upload(instances);
			instanceDataMap.isMapChanged = false;
		}
		if (lightDataMap.isMapChanged && lightDataMap.size() != 0)
		{
			std::vector<LightData*> lightData;
			for (vtxID lightID : lightDataMap)
			{
				if (lightDataMap[lightID].getHostImage().use) // we only add the lights that are marked as lights
				{
					LightData* light = lightDataMap[lightID].getDeviceImage();
					lightData.push_back(light);
				}
			}
			if(!lightData.empty())
			{
				launchParamsData.editableHostImage().lights = launchParamsData.resourceBuffers.lightsDataBuffer.upload(lightData);
			}
			else
			{
				launchParamsData.editableHostImage().lights = nullptr;
			}
			launchParamsData.editableHostImage().numberOfLights = lightData.size();
			lightDataMap.isMapChanged = false;
		}
		if (materialDataMap.isMapChanged)
		{
			materialDataMap.isMapChanged = false;
		}

		if (isFrameIdUpdated)
		{
			// TEST I don't really want to upload the whole launchparams just because I updated the frame count
			// Here I ma trying to check if it is the first time I upload the frame count, in that case, launch params needs to store the proper device ptr
			// However, if it is not the first time, since frameID will always have the same sizeof(int) the device ptr will be the same
			// and I shouldn't need to update the launch params
			launchParamsData.resourceBuffers.frameIdBuffer.upload(frameId);
			if (!launchParamsData.getHostImage().frameID || launchParamsData.getHostImage().frameID != launchParamsData.resourceBuffers.frameIdBuffer.castedPointer<int>())
			{
				launchParamsData.editableHostImage().frameID = launchParamsData.resourceBuffers.frameIdBuffer.castedPointer<int>();
			}
			isFrameIdUpdated = false;
		}
		
		if(debugData.isDirty)
		{
			launchParamsData.editableHostImage().debugData = debugData.getDeviceImage();
		}


	}
	void DeviceDataCoordinator::incrementFrameIteration()
	{
		frameId++;
		isFrameIdUpdated = true;
	}
	void DeviceDataCoordinator::reset()
	{
		optixInstances.clear();
		instanceDataMap.clear();
		geometryDataMap.clear();
		materialDataMap.clear();
		textureDataMap.clear();
		bsdfDataMap.clear();
		lightProfileDataMap.clear();
		lightDataMap.clear();

		// The following data currently does not need a map
		// However if we want to support multiple renderers (Viewports) these should be unique to each viewport, we can map them by rendererID
		noiseComputationData = DeviceData<DummyHostImage, NoiseComputationBuffers>();
		frameBufferData = DeviceData<FrameBufferData, FrameBufferBuffers>();
		launchParamsData = DeviceData<LaunchParams, LaunchParamsBuffers>() ;
		workQueuesData = DeviceData<QueuesData, WorkQueueBuffers>();
		networkInterfaceData = DeviceData<DummyHostImage, NetworkInterfaceBuffer>();

		prevWorkQueueInfo = PrevWorkQueueInfo();
		prevNetworkInterfaceInfo = PrevNetworkInterfaceInfo();

		frameId = 0;
		isFrameIdUpdated = true;
	}
}


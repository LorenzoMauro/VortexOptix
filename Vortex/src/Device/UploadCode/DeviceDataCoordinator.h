#pragma once
#include <optix_types.h>
#include "DeviceDataMap.h"
#include "Core/VortexID.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Scene/NodeTypes.h"
#include "Scene/Scene.h"
#include "Scene/SceneIndexManager.h"
#include "Device/DevicePrograms/DebugDownloadData.h"
#define onDeviceData vtx::device::DeviceDataCoordinator::getInstance()

namespace vtx
{
	namespace graph
	{
		class Instance;
		class Transform;
		class Group;
		class Mesh;
		class Material;
		class Camera;
		class Renderer;
		class Texture;
		class BsdfMeasurement;
		class LightProfile;
		class MeshLight;
		class EnvironmentLight;
	}
}

namespace vtx::device
{
	using DummyHostImage = int;
	using DummyResourceBuffer = int;

	struct DeviceDataCoordinator {

		static DeviceDataCoordinator* getInstance();

		DeviceDataCoordinator(const DeviceDataCoordinator&) = delete;             // Disable copy constructor
		DeviceDataCoordinator& operator=(const DeviceDataCoordinator&) = delete;  // Disable assignment operator
		DeviceDataCoordinator(DeviceDataCoordinator&&) = delete;                  // Disable move constructor
		DeviceDataCoordinator& operator=(DeviceDataCoordinator&&) = delete;       // Disable move assignment operator

		void sync();

		void syncNode(const std::shared_ptr<graph::Instance>& instance);
		void syncNode(const std::shared_ptr<graph::Transform>& transform);
		void syncNode(const std::shared_ptr<graph::Group>& group);
		void syncNode(const std::shared_ptr<graph::Mesh>& mesh);
		void syncNode(const std::shared_ptr<graph::Material>& material);
		void syncNode(const std::shared_ptr<graph::Camera>& camera);
		void syncNode(const std::shared_ptr<graph::Renderer>& renderer);
		void syncNode(const std::shared_ptr<graph::Texture>& textureNode);
		void syncNode(const std::shared_ptr<graph::BsdfMeasurement>& bsdfMeasurementNode);
		void syncNode(const std::shared_ptr<graph::LightProfile>& lightProfile);
		void syncNode(const std::shared_ptr<graph::MeshLight>& lightNode);
		void syncNode(const std::shared_ptr<graph::EnvironmentLight>& lightNode);

		template <typename T, typename B>
		void cleanDeletedNodeOfType(DeviceDataMap<T, B>& dataMap, graph::NodeType type)
		{
			std::vector<math::vec2ui> deletedIds = graph::Scene::getSim()->getDeletedNodesByType(type);
			for (math::vec2ui& ids : deletedIds)
			{
				vtxID id = ids.x;
				if ((type) == graph::NT_INSTANCE)
				{
					id = ids.y;
				}
				VTX_WARN("Node type {} with id {} has been deleted, removing from deviceDataMap", graph::nodeNames[type], id);
				dataMap.erase(id);
				if ((type) == graph::NT_INSTANCE)
				{
					optixInstances.erase(ids.y);
					launchParamsData.editableHostImage().topObject = 0;
				}
				if ((type) == graph::NT_ENV_LIGHT)
				{
					launchParamsData.editableHostImage().envLight = nullptr;
				}
			}
			graph::Scene::getSim()->cleanDeletedNodesByType(type);
		}

		void cleanDeletedNodes();

		void finalize();

		void incrementFrameIteration();

		void reset();

		std::map<vtxID, OptixInstance>                       optixInstances;
		DeviceDataMap<InstanceData, InstanceBuffers>         instanceDataMap;
		DeviceDataMap<GeometryData, GeometryBuffers>         geometryDataMap;
		DeviceDataMap<MaterialData, MaterialBuffers>         materialDataMap;
		DeviceDataMap<TextureData, TextureBuffers>           textureDataMap;
		DeviceDataMap<BsdfData, BsdfBuffers>                 bsdfDataMap;
		DeviceDataMap<LightProfileData, LightProfileBuffers> lightProfileDataMap;
		DeviceDataMap<LightData, LightBuffers>               lightDataMap;

		// The following data currently does not need a map
		// However if we want to support multiple renderers (Viewports) these should be unique to each viewport, we can map them by rendererID
		DeviceData<DummyHostImage, NoiseComputationBuffers> noiseComputationData;
		DeviceData<FrameBufferData, FrameBufferBuffers> frameBufferData;
		DeviceData<LaunchParams, LaunchParamsBuffers> launchParamsData;

		DeviceData<DebugData, DebugDataBuffer> debugData;

		struct PrevWorkQueueInfo
		{
			bool wasAllocated = false;
			int numberOfPixels = 0;
			int maxBounces = 0;
		};

		DeviceData<QueuesData, WorkQueueBuffers> workQueuesData;
		PrevWorkQueueInfo prevWorkQueueInfo;

		struct PrevNetworkInterfaceInfo
		{
			bool                      wasAllocated = false;
			int                       numberOfPixels = 0;
			int                       maxDatasetSize = 0;
			int                       maxDepth = 0;
			network::config::DistributionType distributionType = network::config::D_COUNT;
			int                       mixtureSize = 0;
		};

		DeviceData<DummyHostImage, NetworkInterfaceBuffer> networkInterfaceData;
		PrevNetworkInterfaceInfo prevNetworkInterfaceInfo;

		int										frameId = 0;
		bool									isFrameIdUpdated = true;

	private:
		~DeviceDataCoordinator() = default;
		DeviceDataCoordinator() = default;
	};

}

#pragma once
#include <memory>
#include <set>
#include "SceneIndexManager.h"
#include "Core/VortexID.h"

namespace vtx::graph
{
	class Renderer;
	class Group;
	class Instance;


	class Scene
	{
	public:
		static Scene* get();

		static std::shared_ptr<SceneIndexManager> getSim();
		void                                      init();

		Scene(const Scene&)            = delete; // Disable copy constructor
		Scene& operator=(const Scene&) = delete; // Disable assignment operator
		Scene(Scene&&)                 = delete; // Disable move constructor
		Scene& operator=(Scene&&)      = delete; // Disable move assignment operator

		std::set<vtxID>                     getSelectedInstancesIds() const;
		std::set<std::shared_ptr<Instance>> getSelectedInstances() const;
		void                                addNodesToSelection(const std::set<vtxID>& selected);
		void                                removeNodesToSelection(const std::set<vtxID>& selected);
		void                                removeInstancesFromSelection();
		void                                setSelected(const std::set<vtxID>& selected);
		std::set<vtxID>                     getSelected() const;

		std::shared_ptr<Group>    sceneRoot;
		std::shared_ptr<Renderer> renderer;

	private:
		Scene();
		~Scene() = default;

		std::set<vtxID>                    selectedIds;
		std::shared_ptr<SceneIndexManager> sim; // Shared Pointer so that it can be passed to the nodes and make it outlive the scene to be available to the nodes destructor
	};
}

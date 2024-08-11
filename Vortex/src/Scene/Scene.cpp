#include "Scene.h"
#include "Utility/Operations.h"
#include "Graph.h"
#include "Utility/ModelLoader.h"

namespace vtx::graph
{
	Scene* Scene::get()
	{
		static Scene scene;
		return &scene;
	}

	std::shared_ptr<SceneIndexManager> Scene::getSim() { return get()->sim; }

	void Scene::init()
	{
		sceneRoot = std::make_shared<Group>();
		sim->record(sceneRoot);
		renderer = std::make_shared<Renderer>();
		sim->record(renderer);
		renderer->sceneRoot = sceneRoot;
	}

	std::set<vtxID> fetchInstances(vtxID id)
	{
		const std::shared_ptr<graph::SceneIndexManager> sim = Scene::getSim();
		if (sim->nodeTypeFromUID(id) == NT_INSTANCE)
		{
			return {id};
		}
		if (sim->nodeTypeFromUID(id) == NT_GROUP)
		{
			std::set<vtxID>              instances;
			const std::shared_ptr<Group> group = sim->getNode<Group>(id);
			for (const std::shared_ptr<Node>& child : group->getChildren())
			{
				if (child->getType() == NT_INSTANCE)
				{
					instances.insert(child->getUID());
				}
				else if (child->getType() == NT_GROUP)
				{
					const std::set<vtxID> childInstances = fetchInstances(child->getUID());
					instances.insert(childInstances.begin(), childInstances.end());
				}
			}
			return instances;
		}
		return {};
	}

	std::set<vtxID> Scene::getSelectedInstancesIds() const
	{
		std::set<vtxID> selectedInstances;
		for (auto& id : selectedIds)
		{
			const std::set<vtxID> instances = fetchInstances(id);
			if (!instances.empty())
			{
				selectedInstances.insert(instances.begin(), instances.end());
			}
		}
		return selectedInstances;
	}

	std::set<std::shared_ptr<Instance>> Scene::getSelectedInstances() const
	{
		const std::set<vtxID>               selectedInstanceIds = getSelectedInstancesIds();
		std::set<std::shared_ptr<Instance>> selectedInstances;
		for (const vtxID& instanceID : selectedInstanceIds)
		{
			const std::shared_ptr<Instance>& instance = sim->getNode<Instance>(instanceID);
			selectedInstances.insert(instance);
		}
		return selectedInstances;
	}

	void Scene::addNodesToSelection(const std::set<vtxID>& selected)
	{
		if (selected.empty())
		{
			return;
		}
		selectedIds.insert(selected.begin(), selected.end());
	}

	void Scene::removeNodesToSelection(const std::set<vtxID>& selected)
	{
		for (const vtxID id : selected)
		{
			selectedIds.erase(id);
		}
	}

	void Scene::removeInstancesFromSelection()
	{
		std::set<vtxID> newSelection;
		for (const vtxID id : selectedIds)
		{
			if (sim->nodeTypeFromUID(id) != NT_INSTANCE)
			{
				newSelection.insert(id);
			}
		}
		setSelected(newSelection);
	}

	void Scene::setSelected(const std::set<vtxID>& selected)
	{
		selectedIds = selected;
	}

	std::set<vtxID> Scene::getSelected() const { return selectedIds; }

	Scene::Scene()
	{
		sim = std::make_shared<SceneIndexManager>();
	}
}

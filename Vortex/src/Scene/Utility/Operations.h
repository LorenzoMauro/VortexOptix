#pragma once
#include <memory>
#include "Scene/SceneIndexManager.h"
#include "Core/Math.h"
#include "Scene/Scene.h"

namespace vtx::graph
{
	namespace shader
	{
        class Material;
        class DiffuseReflection;
        class Texture;
        class MaterialSurface;
    }

	struct VertexAttributes;
	class Group;
	class Node;
	class Mesh;
	struct TransformAttribute;

    class Scene;
}

namespace vtx::ops
{
	template <typename T, typename... Ts>
	std::shared_ptr<T> createNode(Ts... optionalArgs)
	{
        static_assert(std::is_base_of_v<graph::Node, T>, "Pushed type is not subclass of Node!");
        std::shared_ptr<T> node = std::make_shared<T>(optionalArgs...);
        graph::Scene::getSim()->record(node);
        return node;
    }

	template <typename T, typename... Ts>
	std::shared_ptr<T> createNodeAndRemoveSIMReferences(Ts... optionalArgs)
	{
        static_assert(std::is_base_of_v<graph::Node, T>, "Pushed type is not subclass of Node!");
        std::shared_ptr<T> node = std::make_shared<T>(optionalArgs...);
        graph::Scene::getSim()->record(node);
        graph::Scene::getSim()->removeNodeReference(node->getUID(), node->getTypeID(), node->getType(), false);
        return node;
    }


    void restartRender();
    // A simple unit cube built from 12 triangles.
    std::shared_ptr<graph::Mesh> createBox(float sideLength=2.0f);

    std::shared_ptr<graph::Camera> standardCamera();

	std::shared_ptr<graph::Mesh> createPlane(float width = 2.0f, float height = 2.0f);

    void updateMaterialSlots(std::shared_ptr<graph::Mesh> mesh, int removedSlot);

    float gaussianFilter(const float* rgba,
                         const unsigned int width,
                         const unsigned int height,
                         const unsigned int x,
                         const unsigned int y,
                         const bool isSpherical);

    std::shared_ptr <graph::shader::ImportedNode> createPbsdfGraph();

    void computeFaceAttributes(const std::shared_ptr<graph::Mesh>& mesh);

    void computeVertexNormals(std::shared_ptr<graph::Mesh> mesh);

    void computeVertexTangentSpace(const std::shared_ptr<graph::Mesh>& mesh);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////// Some Comodity Functions for hard coded scenes /////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    std::shared_ptr<graph::Group> simpleScene01();

    std::tuple<std::shared_ptr<graph::Group>, std::shared_ptr<graph::Camera>> importedScene();
}
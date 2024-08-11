#include "Operations.h"

#include "ModelLoader.h"
#include "Core/Math.h"
#include "Serialization/Serializer.h"
#include "MDL/ShaderVisitor.h"
#include "Scene/Graph.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::ops
{
	using namespace graph;

	void restartRender()
	{
		const std::vector<std::shared_ptr<graph::Renderer>>& renderers = graph::Scene::getSim()->getAllNodeOfType<graph::Renderer>(graph::NT_RENDERER);

		for (const std::shared_ptr<graph::Renderer>& renderer : renderers)
		{
			renderer->restart();
		}
	}

	// A simple unit cube built from 12 triangles.
	std::shared_ptr<graph::Mesh> createBox(float sideLength)
	{
		VTX_INFO("Creating Box");
		std::shared_ptr<Mesh> mesh = createNode<Mesh>();

		const float left   = -sideLength / 2.0f;
		const float right  = sideLength / 2.0f;
		const float bottom = -sideLength / 2.0f;
		const float top    = sideLength / 2.0f;
		const float back   = -sideLength / 2.0f;
		const float front  = sideLength / 2.0f;

		VertexAttributes attrib;

		// Left.
		attrib.tangent = math::vec3f(0.0f, 0.0f, 1.0f);
		attrib.normal  = math::vec3f(-1.0f, 0.0f, 0.0f);

		attrib.position = math::vec3f(left, bottom, back);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, bottom, front);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, top, front);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, top, back);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		// Right.
		attrib.tangent = math::vec3f(0.0f, 0.0f, -1.0f);
		attrib.normal  = math::vec3f(1.0f, 0.0f, 0.0f);

		attrib.position = math::vec3f(right, bottom, front);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, bottom, back);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, back);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, front);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		// Back.  
		attrib.tangent = math::vec3f(-1.0f, 0.0f, 0.0f);
		attrib.normal  = math::vec3f(0.0f, 0.0f, -1.0f);

		attrib.position = math::vec3f(right, bottom, back);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, bottom, back);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, top, back);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, back);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		// Front.
		attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
		attrib.normal  = math::vec3f(0.0f, 0.0f, 1.0f);

		attrib.position = math::vec3f(left, bottom, front);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, bottom, front);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, front);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, top, front);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		// Bottom.
		attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
		attrib.normal  = math::vec3f(0.0f, -1.0f, 0.0f);

		attrib.position = math::vec3f(left, bottom, back);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, bottom, back);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, bottom, front);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, bottom, front);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		// Top.
		attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
		attrib.normal  = math::vec3f(0.0f, 1.0f, 0.0f);

		attrib.position = math::vec3f(left, top, front);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, front);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(right, top, back);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = math::vec3f(left, top, back);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		FaceAttributes faceAttrib;
		for (unsigned int i = 0; i < 6; ++i)
		{
			const unsigned int idx = i * 4; // Four Mesh->vertices per box face.

			mesh->indices.push_back(idx);
			mesh->indices.push_back(idx + 1);
			mesh->indices.push_back(idx + 2);
			mesh->faceAttributes.push_back(faceAttrib);

			mesh->indices.push_back(idx + 2);
			mesh->indices.push_back(idx + 3);
			mesh->indices.push_back(idx);
			mesh->faceAttributes.push_back(faceAttrib);
		}
		//mesh->status.hasNormals = true;
		//mesh->status.hasTangents = true;
		return mesh;
	}

	std::shared_ptr<graph::Camera> standardCamera()
	{
		std::shared_ptr<Camera> camera = ops::createNode<Camera>();
		camera->transform->rotateDegree(math::xAxis, 90.0f);
		camera->transform->rotateDegree(camera->horizontal, -15.0f);
		camera->transform->translate(math::yAxis, -2.0f*3);
		camera->transform->rotateDegree(math::zAxis, -90.0f);
		camera->transform->translate(math::zAxis, 2.0f);
		camera->updateDirections();
		return camera;
	}

	//    // A simple unit cube built from 12 triangles.
	//    std::shared_ptr<Mesh> createBox()
	//    {
	//        VTX_INFO("Creating Box");
	//        std::shared_ptr<Mesh> mesh = createNode<Mesh>();
	//
	//        const float left = -1.0f;
	//        const float right = 1.0f;
	//        const float bottom = -1.0f;
	//        const float top = 1.0f;
	//        const float back = -1.0f;
	//        const float front = 1.0f;
	//
	//        VertexAttributes attrib;
	//
	//        // Left.
	//        //attrib.tangent = math::vec3f(0.0f, 0.0f, 1.0f);
	//        //attrib.normal = math::vec3f(-1.0f, 0.0f, 0.0f);
	//
	//        attrib.position = math::vec3f(left, bottom, back);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, bottom, front);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, top, front);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, top, back);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        // Right.
	//        //attrib.tangent = math::vec3f(0.0f, 0.0f, -1.0f);
	//        //attrib.normal = math::vec3f(1.0f, 0.0f, 0.0f);
	//
	//        attrib.position = math::vec3f(right, bottom, front);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, bottom, back);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, back);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, front);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        // Back.  
	//        //attrib.tangent = math::vec3f(-1.0f, 0.0f, 0.0f);
	//        //attrib.normal = math::vec3f(0.0f, 0.0f, -1.0f);
	//
	//        attrib.position = math::vec3f(right, bottom, back);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, bottom, back);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, top, back);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, back);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        // Front.
	//        //attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
	//        //attrib.normal = math::vec3f(0.0f, 0.0f, 1.0f);
	//
	//        attrib.position = math::vec3f(left, bottom, front);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, bottom, front);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, front);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, top, front);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        // Bottom.
	//        //attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
	//        //attrib.normal = math::vec3f(0.0f, -1.0f, 0.0f);
	//
	//        attrib.position = math::vec3f(left, bottom, back);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, bottom, back);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, bottom, front);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, bottom, front);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        // Top.
	//attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
	//        attrib.normal = math::vec3f(0.0f, 1.0f, 0.0f);
	//
	//        attrib.position = math::vec3f(left, top, front);
	//        attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, front);
	//        attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(right, top, back);
	//        attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        attrib.position = math::vec3f(left, top, back);
	//        attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
	//        mesh->vertices.push_back(attrib);
	//
	//        FaceAttributes faceAttrib;
	//        for (unsigned int i = 0; i < 6; ++i)
	//        {
	//            const unsigned int idx = i * 4; // Four Mesh->vertices per box face.
	//
	//            mesh->indices.push_back(idx);
	//            mesh->indices.push_back(idx + 1);
	//            mesh->indices.push_back(idx + 2);
	//            mesh->faceAttributes.push_back(faceAttrib);
	//
	//            mesh->indices.push_back(idx + 2);
	//            mesh->indices.push_back(idx + 3);
	//            mesh->indices.push_back(idx);
	//            mesh->faceAttributes.push_back(faceAttrib);
	//        }
	//        computeTangents(mesh->vertices, mesh->indices);
	//        return mesh;
	//    }

	std::shared_ptr<graph::Mesh> createPlane(float width, float height)
	{
		VTX_INFO("Creating Plane");
		std::shared_ptr<graph::Mesh> mesh = createNode<graph::Mesh>();

		mesh->vertices.clear();
		mesh->indices.clear();

		math::vec3f corner;

		VertexAttributes attrib;

		// Positive z-axis is the geometry normal, create geometry on the xy-plane.
		corner = math::vec3f(-1.0f, -1.0f, 0.0f); // Lower left corner of the plane. texcoord (0.0f, 0.0f).

		attrib.tangent = math::vec3f(1.0f, 0.0f, 0.0f);
		attrib.normal  = math::vec3f(0.0f, 0.0f, 1.0f);

		attrib.position = corner + math::vec3f(0.0f, 0.0f, 0.0f);
		attrib.texCoord = math::vec3f(0.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = corner + math::vec3f(2.0f, 0.0f, 0.0f);
		attrib.texCoord = math::vec3f(1.0f, 0.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = corner + math::vec3f(0.0f, 2.0f, 0.0f);
		attrib.texCoord = math::vec3f(0.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		attrib.position = corner + math::vec3f(2.0f, 2.0f, 0.0f);
		attrib.texCoord = math::vec3f(1.0f, 1.0f, 0.0f);
		mesh->vertices.push_back(attrib);

		FaceAttributes faceAttrib;
		faceAttrib.materialSlotId = 0;
		//faceAttrib.normal = math::vec3f{ 0.0f, 0.0f, 1.0f };
		//faceAttrib.tangent = math::vec3f{ 1.0f, 0.0f, 0.0f };
		//faceAttrib.bitangent = math::vec3f{ 0.0f, 1.0f, 0.0f };

		mesh->indices.push_back(0);
		mesh->indices.push_back(1);
		mesh->indices.push_back(3);
		mesh->faceAttributes.push_back(faceAttrib);

		mesh->indices.push_back(3);
		mesh->indices.push_back(2);
		mesh->indices.push_back(0);
		mesh->faceAttributes.push_back(faceAttrib);

		//mesh->status.hasFaceAttributes = true;
		//mesh->status.hasNormals = true;
		//mesh->status.hasTangents = true;

		return mesh;
	}

	void updateMaterialSlots(std::shared_ptr<graph::Mesh> mesh, const int removedSlot)
	{
		for (FaceAttributes& face : mesh->faceAttributes)
		{
			if (face.materialSlotId >= removedSlot && face.materialSlotId != 0)
			{
				face.materialSlotId += -1;
			}
		}
	}

	float gaussianFilter(const float* rgba, const unsigned int width, const unsigned int height, const unsigned int x,
						 const unsigned int y, const bool isSpherical)
	{
		// Lookup is repeated in x and clamped to edge in y.
		unsigned int left;
		unsigned int right;
		unsigned int bottom = (0 < y) ? y - 1 : y;          // clamp
		unsigned int top    = (y < height - 1) ? y + 1 : y; // clamp

		// Match the filter to the texture object wrap setup for spherical and rectangular emission textures.
		if (isSpherical) // Spherical environment light 
		{
			left  = (0 < x) ? x - 1 : width - 1; // repeat
			right = (x < width - 1) ? x + 1 : 0; // repeat
		}
		else // Rectangular area light 
		{
			left  = (0 < x) ? x - 1 : x;         // clamp
			right = (x < width - 1) ? x + 1 : x; // clamp
		}


		// Center
		const float* p         = rgba + (width * y + x) * 4;
		float        intensity = (p[0] + p[1] + p[2]) * 0.619347f;

		// 4-neighbours
		p       = rgba + (width * bottom + x) * 4;
		float f = p[0] + p[1] + p[2];
		p       = rgba + (width * y + left) * 4;
		f += p[0] + p[1] + p[2];
		p = rgba + (width * y + right) * 4;
		f += p[0] + p[1] + p[2];
		p = rgba + (width * top + x) * 4;
		f += p[0] + p[1] + p[2];
		intensity += f * 0.0838195f;

		// 8-neighbours corners
		p = rgba + (width * bottom + left) * 4;
		f = p[0] + p[1] + p[2];
		p = rgba + (width * bottom + right) * 4;
		f += p[0] + p[1] + p[2];
		p = rgba + (width * top + left) * 4;
		f += p[0] + p[1] + p[2];
		p = rgba + (width * top + right) * 4;
		f += p[0] + p[1] + p[2];
		intensity += f * 0.0113437f;

		return intensity / 3.0f;
	}

	void computeFaceAttributes(const std::shared_ptr<Mesh>& mesh)
	{
		//const std::vector<VertexAttributes>& vertices = mesh->vertices;
		//const std::vector<vtxID>& indices = mesh->indices;
		//// Initialize normals, tangents, and bitangents to zero

		//for (size_t i = 0; i < indices.size(); i += 3)
		//{
		//    const int faceIndex = i / 3;
		//    FaceAttributes& faceAttr = mesh->faceAttributes[faceIndex];

		//    // Get triangle mesh->vertices
		//    const VertexAttributes& v0 = vertices[indices[i]];
		//    const VertexAttributes& v1 = vertices[indices[i + 1]];
		//    const VertexAttributes& v2 = vertices[indices[i + 2]];

		//    // Compute edges
		//    math::vec3f e1 = v1.position - v0.position;
		//    math::vec3f e2 = v2.position - v0.position;

		//    // Compute face normal
		//    faceAttr.normal = cross(e1, e2);
		//    faceAttr.normal = math::normalize(faceAttr.normal);

		//    // Update face attributes
		//    //FaceAttributes faceAttr;

		//    // Compute UV deltas
		//    math::vec3f deltaUv1 = v1.texCoord - v0.texCoord;
		//    math::vec3f deltaUv2 = v2.texCoord - v0.texCoord;

		//    // Compute tangent and bitangent
		//    float f = 1.0f / (deltaUv1.x * deltaUv2.y - deltaUv2.x * deltaUv1.y);
		//    faceAttr.tangent = (e1 * deltaUv2.y - e2 * deltaUv1.y) * f;
		//    faceAttr.bitangent = (e2 * deltaUv1.x - e1 * deltaUv2.x) * f;

		//    faceAttr.tangent = math::normalize(faceAttr.tangent);
		//    faceAttr.bitangent = math::normalize(faceAttr.bitangent);
		//}
	}

	void computeVertexNormals(std::shared_ptr<Mesh> mesh)
	{
		//VTX_INFO("Computing vertex normals Mesh {}", mesh->getUID());
		//std::vector<VertexAttributes>& vertices = mesh->vertices;
		//std::vector<vtxID>& indices = mesh->indices;
		//// Initialize normals, tangents, and bitangents to zero
		//for (auto& vertex : mesh->vertices)
		//{
		//    vertex.normal = math::vec3f(0.0f, 0.0f, 0.0f);
		//}

		//for (size_t i = 0; i < indices.size(); i += 3)
		//{
		//    const int                   faceIndex = i / 3;
		//    const FaceAttributes& faceAttr = mesh->faceAttributes[faceIndex];

		//    // Get triangle mesh->vertices
		//    VertexAttributes& v0 = vertices[indices[i]];
		//    VertexAttributes& v1 = vertices[indices[i + 1]];
		//    VertexAttributes& v2 = vertices[indices[i + 2]];
		//    // Compute edges
		//    math::vec3f e1 = v1.position - v0.position;
		//    math::vec3f e2 = v2.position - v0.position;

		//    // Compute face normal
		//    math::vec3f normal = cross(e1, e2);
		//    normal = math::normalize(normal);

		//    // Accumulate normals, tangents, and bitangents at each vertex
		//    v0.normal += normal;
		//    v1.normal += normal;
		//    v2.normal += normal;
		//}

		//// Normalize and orthogonalize vertex normals and tangents
		//for (auto& vertex : vertices)
		//{
		//    vertex.normal = math::normalize(vertex.normal);
		//}
		//mesh->status.hasNormals = true;
	}

	void computeVertexTangentSpace(const std::shared_ptr<Mesh>& mesh)
	{
		VTX_INFO("Computing vertex tangents Space for Mesh {}", mesh->getUID());
		std::vector<VertexAttributes>& vertices = mesh->vertices;
		std::vector<vtxID>& indices = mesh->indices;
		// Initialize normals, tangents, and bitangents to zero
		for (auto& vertex : vertices)
		{
			vertex.normal = math::vec3f(0.0f, 0.0f, 0.0f);
			vertex.tangent = math::vec3f(0.0f, 0.0f, 0.0f);
			vertex.bitangent = math::vec3f(0.0f, 0.0f, 0.0f);
		}

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			// Get triangle mesh->vertices
			VertexAttributes& v0 = vertices[indices[i]];
			VertexAttributes& v1 = vertices[indices[i + 1]];
			VertexAttributes& v2 = vertices[indices[i + 2]];


			// Compute edges
			math::vec3f e1 = v1.position - v0.position;
			math::vec3f e2 = v2.position - v0.position;

			// Compute face normal
			math::vec3f normal = cross(e1, e2);
			normal = math::normalize(normal);

			// Update face attributes
			//FaceAttributes faceAttr;

			// Compute UV deltas
			math::vec3f deltaUv1 = v1.texCoord - v0.texCoord;
			math::vec3f deltaUv2 = v2.texCoord - v0.texCoord;

			if (
				deltaUv1 == deltaUv2 ||
				math::isZero(deltaUv1) || math::isZero(deltaUv2) ||
				math::isZero(cross(deltaUv1, deltaUv2)))
			{
				deltaUv1 = math::vec3f(1.0f, 0.0f, 0.0f);
				deltaUv2 = math::vec3f(0.0f, 1.0f, 0.0f);
			}

			// Compute tangent and bitangent
			float       determinant = deltaUv1.x * deltaUv2.y - deltaUv1.y * deltaUv2.x;
			float       f           = 1.0f / determinant;
			math::vec3f tangent     = (e1 * deltaUv2.y - e2 * deltaUv1.y) * f;
			math::vec3f bitangent; // = (e2 * deltaUv1.x - e1 * deltaUv2.x) * f;
			bitangent = cross(normal, tangent);

			if (determinant < 0.0f)
			{
				tangent *= -1.0f;
			}

			float handedness = (dot(cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
			bitangent *= handedness;

			// Accumulate normals, tangents, and bitangents at each vertex
			v0.normal += normal;
			v1.normal += normal;
			v2.normal += normal;

			v0.tangent += tangent;
			v1.tangent += tangent;
			v2.tangent += tangent;

			v0.bitangent += bitangent;
			v1.bitangent += bitangent;
			v2.bitangent += bitangent;
		}

		// Normalize and orthogonalize vertex normals and tangents
		int i = 0;
		for (auto& vertex : vertices)
		{
			math::vec3f& n = vertex.normal;
			math::vec3f& t = vertex.tangent;
			math::vec3f& b = vertex.bitangent;

			if (math::isZero(n))
			{
				n = math::vec3f(0.0f, 0.0f, 1.0f);
				VTX_WARN("Normal Vector was Zero in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}
			if (math::isZero(t))
			{
				t = math::vec3f(1.0f, 0.0f, 0.0f);
				VTX_WARN("Tangent Vector was Zero in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}
			if (math::isZero(b))
			{
				b = math::vec3f(0.0f, 1.0f, 0.0f);
				VTX_WARN("Bitangent Vector was Zero in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}

			n = math::normalize(n);
			if (math::isNan(n))
			{
				n = math::vec3f(0.0f, 0.0f, 1.0f);
				VTX_WARN("Normal Vector was Nan in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}

			t -= n * dot(n, t);
			t = math::normalize(t);

			b -= n * dot(n, b);
			b = math::normalize(b);

			if (math::isNan(t))
			{
				t = math::vec3f(1.0f, 0.0f, 0.0f);
				VTX_WARN("Tangent Vector was Nan in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}
			if (math::isNan(b))
			{
				b = math::vec3f(0.0f, 1.0f, 0.0f);
				VTX_WARN("Bitangent Vector was Nan in mesh Id: {}, vertex index {}", mesh->getUID(), i);
			}

			VTX_ASSERT_CLOSE(!math::isNan(n));
			VTX_ASSERT_CLOSE(!math::isNan(t));
			VTX_ASSERT_CLOSE(!math::isNan(b));

			i++;
		}
		mesh->status.hasTangents = true;
		mesh->status.hasNormals  = true;
	}


	std::shared_ptr<graph::Group> simpleScene01()
	{
		auto sceneRoot = ops::createNode<Group>();
		VTX_INFO("Starting Scene");

		//std::shared_ptr<Material> material1 = ops::createNode<Material>();
		//material1->shader->name = "Stone_Mediterranean";
		//material1->shader->path = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
		//material1->shader->name = "Aluminum";
		//material1->shader->path = "\\vMaterials_2\\Metal\\Aluminum.mdl";
		//material1->shader->name = "bsdf_diffuse_reflection";
		//material1->shader->path = "\\bsdf_diffuse_reflection.mdl";

		const auto principled = ops::createNode<shader::ImportedNode>("E:/Dev/VortexOptix/data/mdl/OmniPBR.mdl",
																	  "OmniPBR");
		principled->printSocketInfos();

		principled->connectInput(ALBEDO_SOCKET, ops::createNode<graph::shader::ColorTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/wlhkfhqv_2K_Albedo.jpg"));
		principled->connectInput(ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/wlhkfhqv_2K_Roughness.jpg"));
		principled->connectInput(NORMALMAP_SOCKET, ops::createNode<graph::shader::NormalTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/wdelddp_2K_Normal.jpg"));
		//principled->sockets["metallic_texture"].parameterInfo.defaultValue = mdl::createTextureConstant("E:/Dev/VortexOptix/data/Textures/xboibga_2K_Normal.jpg");

		const std::shared_ptr<Material> material1 = ops::createNode<Material>();
		material1->materialGraph                  = principled;

		std::shared_ptr<Mesh> cube = ops::createBox();

		std::shared_ptr<Instance> Cube1 = ops::createNode<Instance>();
		Cube1->setChild(cube);
		Cube1->transform->translate(math::xAxis, 2.0f);
		Cube1->addMaterial(material1);
		sceneRoot->addChild(Cube1);
		std::shared_ptr<Material>                          materialEmissive = ops::createNode<Material>();
		const std::shared_ptr<graph::shader::ImportedNode> materialGraph    = ops::createNode<graph::shader::ImportedNode>("\\nvidia\\vMaterials\\AEC\\Lights\\Lights_Emitter.mdl", "naturalwhite_4000k", true);
		materialEmissive->materialGraph                                     = materialGraph;


		std::shared_ptr<Instance> Cube2 = ops::createNode<Instance>();
		Cube2->setChild(cube);
		Cube2->transform->translate(math::yAxis, 2.0f);
		Cube2->addMaterial(material1);

		std::shared_ptr<Instance> Cube3 = ops::createNode<Instance>();
		Cube3->setChild(cube);
		Cube3->transform->rotateDegree(math::xAxis, 45.0f);
		Cube3->transform->translate(math::zAxis, 2.0f);
		Cube3->addMaterial(material1);

		std::shared_ptr<Mesh> plane = ops::createPlane();

		std::shared_ptr<Instance> GroundPlane = ops::createNode<Instance>();
		GroundPlane->setChild(plane);
		GroundPlane->transform->scale(100.0f);
		GroundPlane->transform->translate(math::zAxis, -1.0f);
		GroundPlane->addMaterial(material1);

		std::shared_ptr<Instance> AreaLight = ops::createNode<Instance>();
		AreaLight->setChild(plane);
		AreaLight->transform->rotateDegree(math::xAxis, 180.0f);
		AreaLight->transform->translate(math::zAxis, 7.0f);
		AreaLight->transform->scale(0.5f);
		AreaLight->addMaterial(materialEmissive);


		sceneRoot->addChild(Cube2);
		sceneRoot->addChild(Cube3);
		sceneRoot->addChild(GroundPlane);
		//sceneRoot->addChild(AreaLight);

		std::string envMapPath = getOptions()->dataFolder + "sunset_in_the_chalk_quarry_1k.hdr";
		//std::string envMapPath =  getOptions()->dataFolder  + "studio_small_03_1k.hdr";
		//std::string envMapPath =  getOptions()->dataFolder  + "16x16-in-1024x1024.png";
		//std::string envMapPath =  getOptions()->dataFolder  + "sunset03_EXR.exr";
		//std::string envMapPath =  getOptions()->dataFolder  + "morning07_EXR.exr";
		const std::shared_ptr<EnvironmentLight> envLight = ops::createNode<EnvironmentLight>();
		envLight->envTexture                             = ops::createNode<Texture>(envMapPath);
		sceneRoot->addChild(envLight);
		//std::string envMapPath = getOptions()->dataFolder + "belfast_sunset_puresky_4k.hdr";
		//std::string envMapPath = getOptions()->dataFolder + "mpumalanga_veld_puresky_4k.hdr";
		//std::string envMapPath = getOptions()->dataFolder + "blouberg_sunrise_2_1k.hdr";
		//std::string envMapPath = getOptions()->dataFolder + "qwantani_1k.hdr";

		//std::string envMapPath =  getOptions()->dataFolder  + "studio_small_03_1k.hdr";
		//std::string envMapPath =  getOptions()->dataFolder  + "CheckerBoard.png";
		//std::string envMapPath =  getOptions()->dataFolder  + "sunset03_EXR.exr";
		//std::string envMapPath =  getOptions()->dataFolder  + "morning07_EXR.exr";


		return sceneRoot;
	}


	std::tuple<std::shared_ptr<graph::Group>, std::shared_ptr<graph::Camera>> importedScene()
	{
		enum TestType
		{
			IMPORTED_MATERIALS,
			CREATED_PRINCIPLED,
			IMPORTED_MDL,
			CREATED_MDL
		};

		enum ImportedScene
		{
			BLENDER_TEST,
			SPONZA_OBJ,
			SPONZA_FBX
		};

		enum MdlMaterials
		{
			BSDF_DIFFUSE_REFLECTION,
			STONE_MEDITERRANEAN,
			ALLUMINIUM
		};

		enum EnvironmentMap
		{
			SUNSET_QUARRY,
			SUNSET_BELFAST,
			PURE_SKY_VELD,
			SUNRISE_BLOUBERG,
			PURE_SKY_QWANTANI,
			STUDIO_SMALL_03
		};

		MdlMaterials   mdlMaterial   = STONE_MEDITERRANEAN;
		TestType       testType      = IMPORTED_MATERIALS;
		ImportedScene  importedScene = BLENDER_TEST;
		EnvironmentMap envMap        = PURE_SKY_VELD;


		std::string scenePath;
		switch (importedScene)
		{
			case BLENDER_TEST:
			{
				scenePath = getOptions()->dataFolder + "models/Blender/blenderTest13.gltf";
			}
			break;
			case SPONZA_OBJ:
			{
				scenePath = getOptions()->dataFolder + "models/sponza2/sponza.obj";
			}
			break;
			case SPONZA_FBX:
			{
				scenePath = getOptions()->dataFolder + "models/sponza/FBX 2013/NewSponza_Main_Zup_002.fbx";
			}
			break;
		}

		const auto [sceneRoot, cameras] = importer::importSceneFile(scenePath);

		std::shared_ptr<graph::Camera> camera;
		if (cameras.size() != 0)
		{
			camera = cameras[0];
		}
		else
		{
			camera = standardCamera();
		}

		switch (testType)
		{
			case IMPORTED_MATERIALS: break;
			case CREATED_PRINCIPLED:
			{
				const std::vector<std::shared_ptr<Instance>> instances = graph::Scene::getSim()->getAllNodeOfType<graph::Instance>(NT_INSTANCE);

				const std::shared_ptr<Material> pBsdfMaterial1   = ops::createNode<Material>();
				auto                            principledGraph1 = ops::createNode<graph::shader::PrincipledMaterial>();
				//principledGraph1->connectInput(ALBEDO_SOCKET, ops::createNode<graph::shader::ColorTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/schvfgwp_2K_Albedo.jpg"));
				//principledGraph1->connectInput(ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/schvfgwp_2K_Roughness.jpg"));
				//principledGraph1->connectInput(METALLIC_SOCKET, ops::createNode<graph::shader::MonoTexture>("E:/Dev/VortexOptix/data/models/Blender/textures/schvfgwp_2K_Metalness.jpg"));
				pBsdfMaterial1->materialGraph = principledGraph1;

				for (const auto& instance : instances)
				{
						instance->addMaterial(pBsdfMaterial1, 0);
				}
			}
			break;
			case IMPORTED_MDL:
			{
				const std::vector<std::shared_ptr<Instance>> instances           = graph::Scene::getSim()->getAllNodeOfType<graph::Instance>(NT_INSTANCE);
				const std::shared_ptr<Material>              importedMdlMaterial = ops::createNode<Material>();

				std::string materialName;
				std::string materialPath;

				switch (mdlMaterial)
				{
				case BSDF_DIFFUSE_REFLECTION:
					{
					materialName = "bsdf_diffuse_reflection";
					materialPath = "\\bsdf_diffuse_reflection.mdl";
				}
					break;
				case STONE_MEDITERRANEAN:
				{
					materialName = "Stone_Mediterranean";
					materialPath = "\\vMaterials_2\\Stone\\Stone_Mediterranean.mdl";
				}
				break;
				case ALLUMINIUM:
				{
					materialName = "Aluminum";
					materialPath = "\\vMaterials_2\\Metal\\Aluminum.mdl";
				}
					break;
				}
				const std::shared_ptr<graph::shader::ImportedNode> materialGraph = ops::createNode<graph::shader::ImportedNode>(materialPath, materialName, true);
				importedMdlMaterial->materialGraph                               = materialGraph;

				for (const std::shared_ptr<Instance>& instance : instances)
				{
					instance->addMaterial(importedMdlMaterial, 0);
				}
			}
			break;
		case CREATED_MDL:
			{
				const auto roughnessTextureMono = ops::createNode<graph::shader::MonoTexture>("E:/Dev/VortexOptix/data/Textures/xboibga_2K_Roughness.jpg");
				const auto diffuseTextureColor  = ops::createNode<graph::shader::ColorTexture>("E:/Dev/VortexOptix/data/Textures/xboibga_2K_Albedo.jpg");

				const auto brdfNode = ops::createNode<shader::DiffuseReflection>();
				brdfNode->connectInput("tint", diffuseTextureColor);
				brdfNode->connectInput("roughness", roughnessTextureMono);
				brdfNode->printSocketInfos();

				const auto surfaceNode = ops::createNode<shader::MaterialSurface>();
				surfaceNode->connectInput("scattering", brdfNode);

				const auto materialGraph = ops::createNode<shader::Material>();
				materialGraph->connectInput("surface", surfaceNode);

				const std::vector<std::shared_ptr<Instance>> instances       = graph::Scene::getSim()->getAllNodeOfType<graph::Instance>(NT_INSTANCE);
				const std::shared_ptr<Material>              createdMaterial = ops::createNode<Material>();
				createdMaterial->materialGraph                               = materialGraph;

				for (const std::shared_ptr<Instance>& instance : instances)
				{
					instance->addMaterial(createdMaterial, 0);
				}
			}
			break;
		}

		// Environment Map Settings
		{
			std::string envMapPath;
			switch (envMap)
			{
			case SUNSET_QUARRY: { envMapPath = getOptions()->dataFolder + "Hdri/" + "sunset_in_the_chalk_quarry_1k.hdr"; }
				break;
			case SUNSET_BELFAST: { envMapPath = getOptions()->dataFolder + "Hdri/" + "belfast_sunset_puresky_4k.hdr"; }
				break;
			case PURE_SKY_VELD: { envMapPath = getOptions()->dataFolder + "Hdri/" + "mpumalanga_veld_puresky_4k.hdr"; }
				break;
			case SUNRISE_BLOUBERG: { envMapPath = getOptions()->dataFolder + "Hdri/" + "blouberg_sunrise_2_1k.hdr"; }
				break;
			case PURE_SKY_QWANTANI: { envMapPath = getOptions()->dataFolder + "Hdri/" + "qwantani_1k.hdr"; }
				break;
			case STUDIO_SMALL_03: { envMapPath = getOptions()->dataFolder + "Hdri/" + "studio_small_03_1k.hdr"; }
				break;
			}

			const std::shared_ptr<EnvironmentLight> envLight = ops::createNode<EnvironmentLight>();
			envLight->envTexture                             = ops::createNode<Texture>(envMapPath);
			sceneRoot->addChild(envLight);
		}

		return {sceneRoot, camera};
	}
}

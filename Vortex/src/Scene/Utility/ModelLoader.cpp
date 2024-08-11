#include "ModelLoader.h"

#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"
#include "Scene/Graph.h"
#include "assimp/GltfMaterial.h"

namespace vtx::importer
{
	static aiMatrix4x4 sceneRotationToVTXFrame = aiMatrix4x4(
															 1.0f, 0.0f, 0.0f, 0.0f,
															 0.0f, 1.0f, 0.0f, 0.0f,
															 0.0f, 0.0f, 1.0f, 0.0f,
															 0.0f, 0.0f, 0.0f, 1.0f
															);

	struct GetTextureInput
	{
		aiTextureType type;
		unsigned      index;
	};

	// Types sorted by preferences
	void getTexturePath(const aiMaterial* material, const std::vector<GetTextureInput>& potentialTypes, std::string& returnPath, const std::string& scenePath)
	{
		aiString path;
		for (const auto potential : potentialTypes)
		{
			if (material->GetTexture(potential.type, potential.index, &path) == AI_SUCCESS)
			{
				returnPath = utl::absolutePath(path.C_Str(), scenePath);
				returnPath = utl::replacePercent20WithSpace(returnPath);
				return;
			}
		}
	}

	struct GetValueInput
	{
		const char* pKey;
		int         type;
		int         index;
	};

	void getColorValue(const aiMaterial* material, const std::vector<GetValueInput>& potentialTypes, math::vec3f& color)
	{
		aiColor4D aiColor;
		for (const auto type : potentialTypes)
		{
			if (material->Get(type.pKey, type.type, type.index, aiColor) == AI_SUCCESS)
			{
				color = math::vec3f(aiColor.r, aiColor.g, aiColor.b);
				return;
			}
		}
	}


	void getFloatValue(const aiMaterial* material, const std::vector<GetValueInput>& potentialTypes, float& color)
	{
		ai_real value;
		for (const auto type : potentialTypes)
		{
			if (material->Get(type.pKey, type.type, type.index, value) == AI_SUCCESS)
			{
				color = value;
				return;
			}
		}
	}

	std::shared_ptr<graph::shader::PrincipledMaterial> createPrincipledMaterial(const AssimpMaterialProperties& properties)
	{
		auto principled = ops::createNode<graph::shader::PrincipledMaterial>();

		const std::string name = properties.name;
		if (!name.empty())
		{
			principled->name = "";
			for (size_t i = 0; i < name.size(); ++i)
			{
				if (name[i] == '.')
				{
					principled->name.push_back('_');
				}
				else
				{
					principled->name.push_back(name[i]);
				}
			}
		}
		VTX_INFO("Loading material: {}, old Name {}", principled->name, name);

		// Diffuse
		std::shared_ptr<graph::shader::ColorTexture> diffuse = nullptr;
		if (!properties.diffuse.path.empty())
		{
			diffuse = ops::createNode<graph::shader::ColorTexture>(properties.diffuse.path);
			principled->connectInput(ALBEDO_SOCKET, diffuse);
		}
		else if (properties.diffuse.value != math::vec3f(-1.0f))
		{
			principled->setSocketValue(ALBEDO_SOCKET, mdl::createConstantColor(properties.diffuse.value));
		}

		// Diffuse
		if (!properties.opacity.path.empty())
		{
			const auto alphaChannel = ops::createNode<graph::shader::MonoTexture>(properties.opacity.path);
			alphaChannel->setSocketValue(VF_MONO_TEXTURE_ALPHA_SOCKET, mdl::createConstantBool(true));
			principled->connectInput(OPACITY_SOCKET, alphaChannel);
			principled->setSocketValue(ENABLE_OPACITY_SOCKET, mdl::createConstantBool(true));
			principled->setSocketValue(THIN_WALLED_SOCKET, mdl::createConstantBool(true));
		}
		else if (properties.opacity.value >= 0.0f && properties.opacity.value < 1.0f)
		{
			principled->setSocketValue(ENABLE_OPACITY_SOCKET, mdl::createConstantBool(true));
			principled->setSocketValue(OPACITY_SOCKET, mdl::createConstantFloat(properties.opacity.value));
			principled->setSocketValue(THIN_WALLED_SOCKET, mdl::createConstantBool(true));
		}

		if (!properties.ORM.path.empty())
		{
			const auto ormTexture = ops::createNode<graph::shader::ColorTexture>(properties.ORM.path);
			if (properties.roughness.value != 1.0f)
			{
				principled->setSocketValue(ROUGHNESS_SOCKET, mdl::createConstantFloat(properties.roughness.value));
			}
			else
			{
				const auto roughnessChannel = ops::createNode<graph::shader::GetChannel>(1);
				roughnessChannel->connectInput(VF_GET_COLOR_CHANNEL_COLOR_SOCKET, ormTexture);
				principled->connectInput(ROUGHNESS_SOCKET, roughnessChannel);
			}

			if (properties.metallic.value != 1.0f)
			{
				principled->setSocketValue(METALLIC_SOCKET, mdl::createConstantFloat(properties.metallic.value));
			}
			else
			{
				const auto metallicChannel = ops::createNode<graph::shader::GetChannel>(2);
				metallicChannel->connectInput(VF_GET_COLOR_CHANNEL_COLOR_SOCKET, ormTexture);
				principled->connectInput(METALLIC_SOCKET, metallicChannel);
			}
		}
		else
		{
			// Metallic
			if (!properties.metallic.path.empty())
			{
				principled->connectInput(METALLIC_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.metallic.path));
			}
			else if (properties.metallic.value >= 0.0f)
			{
				principled->setSocketValue(METALLIC_SOCKET, mdl::createConstantFloat(properties.metallic.value));
			}

			// Roughness
			if (!properties.roughness.path.empty())
			{
				principled->connectInput(ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.roughness.path));
			}
			else if (properties.roughness.value >= 0.0f)
			{
				principled->setSocketValue(ROUGHNESS_SOCKET, mdl::createConstantFloat(properties.roughness.value));
			}
		}


		//Normal and Bump
		if (!properties.normal.path.empty() && !properties.bump.path.empty())
		{
			const auto normalMap = ops::createNode<graph::shader::NormalTexture>(properties.normal.path);
			const auto bumpMap   = ops::createNode<graph::shader::BumpTexture>(properties.bump.path);
			const auto mixNormal = ops::createNode<graph::shader::NormalMix>();

			mixNormal->connectInput(VF_MIX_NORMAL_BASE_SOCKET, normalMap);
			mixNormal->connectInput(VF_MIX_NORMAL_LAYER_SOCKET, bumpMap);

			principled->connectInput(NORMALMAP_SOCKET, mixNormal);
		}
		else if (!properties.normal.path.empty())
		{
			const auto normalMap = ops::createNode<graph::shader::NormalTexture>(properties.normal.path);
			principled->connectInput(NORMALMAP_SOCKET, normalMap);
		}
		else if (!properties.bump.path.empty())
		{
			const auto bumpMap = ops::createNode<graph::shader::BumpTexture>(properties.bump.path);
			principled->connectInput(NORMALMAP_SOCKET, bumpMap);
		}

		// Emission

		if (!properties.emissionIntensity.path.empty())
		{
			principled->setSocketValue(ENABLE_EMISSION_SOCKET, mdl::createConstantBool(true));
			principled->connectInput(EMISSION_INTENSITY_SOCKET, ops::createNode<graph::shader::ColorTexture>(properties.emissionIntensity.path));
		}
		else if (properties.emissionIntensity.value >= 1.0f)
		{
			principled->setSocketValue(ENABLE_EMISSION_SOCKET, mdl::createConstantBool(true));
			principled->setSocketValue(EMISSION_INTENSITY_SOCKET, mdl::createConstantFloat(properties.emissionIntensity.value));
		}
		if (!properties.emissionColor.path.empty())
		{
			principled->setSocketValue(ENABLE_EMISSION_SOCKET, mdl::createConstantBool(true));
			principled->connectInput(EMISSION_COLOR_SOCKET, ops::createNode<graph::shader::ColorTexture>(properties.emissionColor.path));
		}
		else if (properties.emissionColor.value != math::vec3f(-1.0f))
		{
			principled->setSocketValue(EMISSION_COLOR_SOCKET, mdl::createConstantColor(properties.emissionColor.value));
			if (properties.emissionColor.value != math::vec3f(0.0f))
			{
				principled->setSocketValue(ENABLE_EMISSION_SOCKET, mdl::createConstantBool(true));
				//HACK this is a hack to compensate for lack of blender to gltf intensity eport
				principled->setSocketValue(EMISSION_INTENSITY_SOCKET, mdl::createConstantFloat(100.0f));
			}
		}

		// ClearCoat
		if (!properties.clearcoatAmount.path.empty())
		{
			principled->connectInput(COAT_AMOUNT_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.clearcoatAmount.path));
		}
		else if (properties.clearcoatAmount.value >= 0.0f)
		{
			principled->setSocketValue(COAT_AMOUNT_SOCKET, mdl::createConstantFloat(properties.clearcoatAmount.value));
		}

		if (!properties.clearcoatRoughness.path.empty())
		{
			principled->connectInput(COAT_ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.clearcoatRoughness.path));
		}
		else if (properties.clearcoatRoughness.value >= 0.0f)
		{
			principled->setSocketValue(COAT_ROUGHNESS_SOCKET, mdl::createConstantFloat(properties.clearcoatRoughness.value));
		}

		if (!properties.clearcoatNormal.path.empty())
		{
			const auto clearcoatNormalMap = ops::createNode<graph::shader::NormalTexture>(properties.clearcoatNormal.path);
			principled->connectInput(COAT_NORMALMAP_SOCKET, clearcoatNormalMap);
		}

		// Transmission
		if (!properties.transmission.path.empty())
		{
			principled->connectInput(TRANSMISSION_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.transmission.path));
		}
		else if (properties.transmission.value >= 0.0f)
		{
			principled->setSocketValue(TRANSMISSION_SOCKET, mdl::createConstantFloat(properties.transmission.value));
		}

		// Sheen
		if (!properties.sheenColor.path.empty())
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->connectInput(SHEEN_TINT_SOCKET, ops::createNode<graph::shader::ColorTexture>(properties.sheenColor.path));
		}
		else if (properties.sheenColor.value != math::vec3f(-1.0f))
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->setSocketValue(SHEEN_TINT_SOCKET, mdl::createConstantColor(properties.sheenColor.value));
		}

		if (!properties.sheenRoughness.path.empty())
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->connectInput(SHEEN_ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.sheenRoughness.path));
		}
		else if (properties.sheenRoughness.value >= 0.0f)
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->setSocketValue(SHEEN_ROUGHNESS_SOCKET, mdl::createConstantFloat(properties.sheenRoughness.value));
		}

		// Anisotropy
		if (properties.anisotropy.value >= 0.0f)
		{
			principled->setSocketValue(ANISOTROPY_SOCKET, mdl::createConstantFloat(abs(properties.anisotropy.value)));
			// Since there's no separate texture or value for anisotropy rotation, you may need to set a default value or use a value based on some condition.
		}


		// Opacity
		if (!properties.sheenColor.path.empty())
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->connectInput(SHEEN_TINT_SOCKET, ops::createNode<graph::shader::ColorTexture>(properties.sheenColor.path));
		}
		else if (properties.sheenColor.value != math::vec3f(-1.0f))
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->setSocketValue(SHEEN_TINT_SOCKET, mdl::createConstantColor(properties.sheenColor.value));
		}

		if (!properties.sheenRoughness.path.empty())
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->connectInput(SHEEN_ROUGHNESS_SOCKET, ops::createNode<graph::shader::MonoTexture>(properties.sheenRoughness.path));
		}
		else if (properties.sheenRoughness.value >= 0.0f)
		{
			principled->setSocketValue(SHEEN_AMOUNT_SOCKET, mdl::createConstantFloat(0.5f));
			principled->setSocketValue(SHEEN_ROUGHNESS_SOCKET, mdl::createConstantFloat(properties.sheenRoughness.value));
		}

		return principled;
	}


	void AssimpMaterialProperties::determineProperties(const aiMaterial* material, std::string scenePath)
	{
		aiString matName;
		material->Get(AI_MATKEY_NAME, matName);

		name = matName.C_Str();
		// Diffuse texture and color
		AI_MATKEY_TEXTURE_DIFFUSE()
			getTexturePath(material, {{aiTextureType_DIFFUSE, 0}, {AI_MATKEY_BASE_COLOR_TEXTURE}}, diffuse.path, scenePath);
		getColorValue(material, {{AI_MATKEY_COLOR_DIFFUSE}, {AI_MATKEY_COLOR_SPECULAR}}, diffuse.value);

		// Ambient Occlusion texture and color
		getTexturePath(material, {{aiTextureType_AMBIENT_OCCLUSION, 0}}, ambientOcclusion.path, scenePath);

		// Roughness texture and value
		getTexturePath(material, {{AI_MATKEY_ROUGHNESS_TEXTURE}, {aiTextureType_SHININESS, 0}}, roughness.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_ROUGHNESS_FACTOR}}, roughness.value); //, {AI_MATKEY_SHININESS}

		// Metallic texture and value
		getTexturePath(material, {{AI_MATKEY_METALLIC_TEXTURE}}, metallic.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_METALLIC_FACTOR}, {AI_MATKEY_REFLECTIVITY}}, metallic.value);

		// MetallicRoughness texture
		getTexturePath(material, {{aiTextureType_UNKNOWN, 0}}, ORM.path, scenePath);

		// Specular texture and value
		getTexturePath(material, {{aiTextureType_SPECULAR, 0}}, specular.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_GLOSSINESS_FACTOR}, {AI_MATKEY_SPECULAR_FACTOR}}, specular.value);


		if (specular.value == -1.0f)
		{
			math::vec3f shininessColor = -1.0f;
			getColorValue(material, {{AI_MATKEY_COLOR_SPECULAR}}, shininessColor);
			if (shininessColor != diffuse.value && shininessColor.x == shininessColor.y && shininessColor.x == shininessColor.z)
			{
				if (shininessColor.x != -1.0f)
				{
					specular.value = shininessColor.x;
				}
			}
		}

		// Normal texture
		getTexturePath(material, {{aiTextureType_NORMAL_CAMERA, 0}, {aiTextureType_NORMALS, 0}}, normal.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_BUMPSCALING}}, normal.value);
		// Bump texture
		getTexturePath(material, {{aiTextureType_HEIGHT, 0}}, bump.path, scenePath);


		// Metallic texture and value
		getTexturePath(material, {{aiTextureType_EMISSION_COLOR, 0}}, emissionColor.path, scenePath);
		getColorValue(material, {{AI_MATKEY_COLOR_EMISSIVE}}, emissionColor.value);

		getTexturePath(material, {{aiTextureType_EMISSIVE, 0}}, emissionIntensity.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_EMISSIVE_INTENSITY}}, emissionIntensity.value);

		//ClearCoat
		getTexturePath(material, {{AI_MATKEY_CLEARCOAT_TEXTURE}}, clearcoatAmount.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_CLEARCOAT_FACTOR}}, clearcoatAmount.value);

		getTexturePath(material, {{AI_MATKEY_CLEARCOAT_ROUGHNESS_TEXTURE}}, clearcoatRoughness.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR}}, clearcoatRoughness.value);

		getTexturePath(material, {{AI_MATKEY_CLEARCOAT_NORMAL_TEXTURE}}, clearcoatNormal.path, scenePath);

		//Transmission
		getTexturePath(material, {{AI_MATKEY_TRANSMISSION_TEXTURE}}, transmission.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_TRANSMISSION_FACTOR}}, transmission.value);

		//Sheen
		getTexturePath(material, {{AI_MATKEY_SHEEN_COLOR_TEXTURE}}, sheenColor.path, scenePath);
		getColorValue(material, {{AI_MATKEY_SHEEN_COLOR_FACTOR}}, sheenColor.value);

		getTexturePath(material, {{AI_MATKEY_CLEARCOAT_ROUGHNESS_TEXTURE}}, sheenRoughness.path, scenePath);
		getFloatValue(material, {{AI_MATKEY_SHEEN_ROUGHNESS_FACTOR}}, sheenRoughness.value);

		//Anisotropy
		getFloatValue(material, {{AI_MATKEY_ANISOTROPY_FACTOR}}, anisotropy.value);

		getFloatValue(material, {{AI_MATKEY_OPACITY}}, opacity.value);
		getTexturePath(material, {{aiTextureType_OPACITY, 0}}, opacity.path, scenePath);

		aiString alphaMode;
		float    opacityValue = 1.0;
		if ((material->Get(AI_MATKEY_GLTF_ALPHAMODE, alphaMode) == AI_SUCCESS && alphaMode == aiString("BLEND"))
			|| (material->Get(AI_MATKEY_OPACITY, opacityValue) == AI_SUCCESS && opacityValue < 1.0))
		{
			opacity.path = opacity.path.empty() ? diffuse.path : opacity.path;
		}
	}

	// Process materials in the aiScene
	std::vector<std::shared_ptr<graph::Material>> processMaterials(const aiScene* scene, std::string scenePath)
	{
		std::vector<std::shared_ptr<graph::Material>> materials;

		for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
		{
			const aiMaterial* aiMat = scene->mMaterials[i];

			// Create a new material in your renderer and set its properties based on aiMat
			auto material = ops::createNode<graph::Material>();
			// Set material properties, e.g., diffuse color, specular color, textures, etc.
			// ...

			AssimpMaterialProperties properties;
			properties.determineProperties(aiMat, scenePath);
			std::shared_ptr<graph::shader::PrincipledMaterial> principled = createPrincipledMaterial(properties);
			// You can now use the extracted variables within the function
			material->materialGraph = principled;

			materials.push_back(material);
		}

		return materials;
	}

	math::vec3f toVec3f(const aiVector3D& vec, SwapType st)
	{
		if (st == SwapType::yToZ)
		{
			return math::vec3f(vec.x, vec.z, vec.y);
		}
		else
		{
			return math::vec3f(vec.x, vec.y, vec.z);
		}
	}

	std::shared_ptr<graph::Mesh> convertAssimpMeshToMeshNode(const aiMesh* aiMesh, SwapType swap)
	{
		// Process vertices
		const auto meshNode = ops::createNode<graph::Mesh>();
		meshNode->vertices.resize(aiMesh->mNumVertices);
		meshNode->status.hasNormals        = aiMesh->HasNormals();
		meshNode->status.hasTangents       = aiMesh->HasTangentsAndBitangents();
		meshNode->status.hasFaceAttributes = aiMesh->HasTangentsAndBitangents();
		VTX_INFO("Mesh {} has normals: {}, has tangents: {}"
				 ,
				 meshNode->name,
				 meshNode->status.hasNormals,
				 meshNode->status.hasTangents
				);
		for (unsigned int i = 0; i < aiMesh->mNumVertices; ++i)
		{
			auto& vertex = meshNode->vertices[i];
			// Positions
			vertex.position = toVec3f(aiMesh->mVertices[i], swap);

			// Normals
			if (meshNode->status.hasNormals)
			{
				vertex.normal = toVec3f(aiMesh->mNormals[i], swap);
			}

			// Tangents
			if (meshNode->status.hasTangents)
			{
				vertex.tangent   = toVec3f(aiMesh->mTangents[i], swap);
				vertex.bitangent = toVec3f(aiMesh->mBitangents[i], swap);
			}

			// Texture coordinates
			if (aiMesh->HasTextureCoords(0))
			{
				vertex.texCoord = math::vec3f(aiMesh->mTextureCoords[0][i].x, aiMesh->mTextureCoords[0][i].y, 0.0f);
			}
		}

		// Handle the winding order if yToZ is true
		if (swap == SwapType::yToZ)
		{
			for (unsigned int i = 0; i < aiMesh->mNumFaces; ++i)
			{
				const aiFace& face = aiMesh->mFaces[i];
				if (face.mNumIndices == 3)
				{
					std::swap(face.mIndices[1], face.mIndices[2]);
				}
			}
		}

		// Process faces and indices
		meshNode->faceAttributes.resize(aiMesh->mNumFaces);
		for (unsigned int i = 0; i < aiMesh->mNumFaces; ++i)
		{
			const aiFace& face = aiMesh->mFaces[i];

			// Store indices for the face
			if (face.mNumIndices != 3)
			{
				std::cout << "Warning: face with " << face.mNumIndices << " indices found. Only triangles are supported." << std::endl;
			}
			else
			{
				for (unsigned int j = 0; j < face.mNumIndices; ++j)
				{
					meshNode->indices.push_back(face.mIndices[j]);
				}

				// Set face attributes (you can update it later based on the material, if needed)
				graph::FaceAttributes faceAttr;
				faceAttr.materialSlotId     = 0;
				meshNode->faceAttributes[i] = faceAttr;
			}
		}

		return meshNode;
	}

	math::vec3f swapZY(const math::vec3f& vec)
	{
		return math::vec3f(vec.x, vec.z, vec.y);
	}

	math::affine3f convertAssimpMatrix(const aiMatrix4x4& aiMatrix, SwapType swap)
	{
		math::affine3f matrix;
		math::vec3f    assimpX(aiMatrix.a1, aiMatrix.b1, aiMatrix.c1);
		math::vec3f    assimpY(aiMatrix.a2, aiMatrix.b2, aiMatrix.c2);
		math::vec3f    assimpZ(aiMatrix.a3, aiMatrix.b3, aiMatrix.c3);

		matrix.l = math::LinearSpace3f(assimpX, assimpY, assimpZ);
		matrix.p = math::vec3f(aiMatrix.a4, aiMatrix.b4, aiMatrix.c4);

		if (swap == SwapType::yToZ)
		{
			math::vec3f scale       = {};
			math::vec3f rotation    = {};
			math::vec3f translation = {};


			math::VectorFromAffine(matrix, translation, scale, rotation);
			if (swap == SwapType::yToZ)
			{
				translation = swapZY(translation);
				rotation    = swapZY(rotation);
				scale       = swapZY(scale);
			}

			matrix = math::affine3f::translate(translation) * math::AffineFromEuler<math::LinearSpace3f>(rotation) * math::affine3f::scale(scale);
		}

		return matrix;
	}

	std::shared_ptr<graph::Instance> processAssimpNode(aiMesh* node, const unsigned assimpMeshId, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials, SwapType swap)
	{
		std::shared_ptr<graph::Mesh> meshNode = nullptr;
		if (meshMap.find(assimpMeshId) != meshMap.end())
		{
			meshNode = graph::Scene::getSim()->getNode<graph::Mesh>(meshMap[assimpMeshId]);
		}
		else
		{
			meshNode = convertAssimpMeshToMeshNode(node, swap);
			meshMap.insert({assimpMeshId, meshNode->getUID()});
		}

		std::shared_ptr<graph::Instance> instanceNode = ops::createNode<graph::Instance>();
		// Set the meshNode as a child of the instanceNode
		instanceNode->setChild(meshNode);
		instanceNode->addMaterial(importedMaterials[node->mMaterialIndex]);
		return instanceNode;
	}

	std::shared_ptr<graph::Node> processAssimpNode(const aiNode* node, const aiScene* scene, std::map<unsigned, vtxID>& meshMap, const std::vector<std::shared_ptr<graph::Material>>& importedMaterials, SwapType swap)
	{
		std::vector<std::shared_ptr<graph::Node>> children;

		// Process node children
		for (unsigned int i = 0; i < node->mNumChildren; ++i)
		{
			children.push_back(processAssimpNode(node->mChildren[i], scene, meshMap, importedMaterials, swap));
		}

		// Process node meshes
		for (unsigned int i = 0; i < node->mNumMeshes; ++i)
		{
			aiMesh*                                aiMesh   = scene->mMeshes[node->mMeshes[i]];
			const std::shared_ptr<graph::Instance> instance = processAssimpNode(aiMesh, node->mMeshes[i], meshMap, importedMaterials, swap);
			children.push_back(instance);
		}

		// If there's only one child and it's a mesh, return the mesh's instance directly.
		if (children.size() == 1 && dynamic_cast<graph::Instance*>(children[0].get()))
		{
			std::shared_ptr<graph::Instance> instance = children[0]->as<graph::Instance>();
			instance->transform->setAffine(convertAssimpMatrix(node->mTransformation, swap));
			return instance;
		}
		else
		{
			auto groupNode = ops::createNode<graph::Group>();
			// Process node transformation
			groupNode->transform->setAffine(convertAssimpMatrix(node->mTransformation, swap));
			for (auto& child : children)
			{
				groupNode->addChild(child);
			}
			return groupNode;
		}
	}

	void processMetadata(const aiScene* scene, const std::string& fileFormat)
	{
		if (scene->mMetaData != nullptr)
		{
			/* int32_t frontAxis = 0; 
			 int32_t frontAxisSign = 1; 
			 int32_t coordAxis = 1; 
			 int32_t coordAxisSign = 1;
			 int32_t upAxis = 2;
			 int32_t upAxisSign = 1;
			 double unitScaleFactor = 0.01;*/
			int32_t frontAxis       = 0;
			int32_t frontAxisSign   = 1;
			int32_t coordAxis       = 2; // Changed from 1 to 2
			int32_t coordAxisSign   = 1;
			int32_t upAxis          = 1; // Changed from 2 to 1
			int32_t upAxisSign      = 1;
			double  unitScaleFactor = 1.0;
			if (fileFormat == "fbx")
			{
				unitScaleFactor = 0.01;
			}
			else if (fileFormat == "obj")
			{
				unitScaleFactor = 0.01;
			}
			else
			{
				unitScaleFactor = 1.0;
			}
			for (unsigned MetadataIndex = 0; MetadataIndex < scene->mMetaData->mNumProperties; ++MetadataIndex)
			{
				VTX_INFO("Metadata: {0}", scene->mMetaData->mKeys[MetadataIndex].C_Str());
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UpAxis") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, upAxis);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata UpAxis");
					}
				}
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UpAxisSign") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, upAxisSign);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata UpAxisSign");
					}
				}
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "FrontAxis") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, frontAxis);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata FrontAxis");
					}
				}
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "FrontAxisSign") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, frontAxisSign);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata FrontAxisSign");
					}
				}
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "CoordAxis") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, coordAxis);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata CoordAxis");
					}
				}
				if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "CoordAxisSign") == 0)
				{
					const bool result = scene->mMetaData->Get<int32_t>(MetadataIndex, coordAxisSign);
					if (!result)
					{
						VTX_WARN("Some Error Occurred Collecting Assimp Metadata CoordAxisSign");
					}
				}
				/*if (strcmp(scene->mMetaData->mKeys[MetadataIndex].C_Str(), "UnitScaleFactor") == 0)
				{
					bool result = scene->mMetaData->Get<double>(MetadataIndex, unitScaleFactor);
					if (!result)
					{
                        float unitScaleFactorFloat;
                        bool result = scene->mMetaData->Get<float>(MetadataIndex, unitScaleFactorFloat);
                        if (!result)
                        {
                            VTX_WARN("Some Error Occurred Collecting Assimp Metadata UnitScaleFactor");
                        }
                        else
                        {
                            unitScaleFactor = (double)unitScaleFactorFloat;
                        }
					}
				}*/
			}

			aiVector3D upVec;
			aiVector3D forwardVec;
			aiVector3D rightVec;

			upVec[upAxis]         = upAxisSign * (float)unitScaleFactor;
			forwardVec[frontAxis] = frontAxisSign * (float)unitScaleFactor;
			rightVec[coordAxis]   = coordAxisSign * (float)unitScaleFactor;

			const aiMatrix4x4 mat(forwardVec.x, forwardVec.y, forwardVec.z, 0.0f,
								  rightVec.x, rightVec.y, rightVec.z, 0.0f,
								  upVec.x, upVec.y, upVec.z, 0.0f,
								  0.0f, 0.0f, 0.0f, 1.0f);

			sceneRotationToVTXFrame = mat;
			//applyTransformationToNode(scene->mRootNode, mat);
			//applyTransformationToLeafNodes(scene->mRootNode, mat);

			//scene->mRootNode->mTransformation = mat;
		}
	}

	std::vector<std::shared_ptr<graph::Camera>> processCameras(const aiScene* scene, SwapType swap)
	{
		std::vector<std::shared_ptr<graph::Camera>> cameras;
		for (unsigned int i = 0; i < scene->mNumCameras; ++i)
		{
			const aiCamera* cam = scene->mCameras[i];

			// Extract camera properties
			float fov      = cam->mHorizontalFOV;
			float aspect   = cam->mAspect;
			float clipNear = cam->mClipPlaneNear;
			float clipFar  = cam->mClipPlaneFar;

			// Position and direction are expressed in the local space of the node
			// the camera is attached to. These will need to be transformed by the
			// node's transform to get them into world space.
			// Get the node that this camera is attached to
			aiNode* node = scene->mRootNode->FindNode(cam->mName);

			// Get the transformation of the node
			//math::affine3f transform = convertAssimpMatrix(node->mTransformation, SwapType::None);
			//math::affine3f rootM = convertAssimpMatrix(scene->mRootNode->mTransformation);
			//math::affine3f finalTransform = rootM * transform;
			// Transform the camera's local vectors into world space
			//aiVector3D posAssimp = cam->mPosition;
			//aiVector3D upAssimp = cam->mUp;
			//aiVector3D lookAssimp = cam->mLookAt;
			//
			//math::vec3f pos (posAssimp.x, posAssimp.y, posAssimp.z);
			//math::vec3f up = math::normalize({ upAssimp.x, upAssimp.y, upAssimp.z });
			//math::vec3f lookAt = math::normalize({lookAssimp.x, lookAssimp.y, lookAssimp.z});
			//math::vec3f horizzontal = math::normalize(cross(lookAt, up));
			//
			//math::vec3f wUp = math::transformVector3F(finalTransform, up);
			//math::vec3f wLookAt = math::transformVector3F(finalTransform, lookAt);
			//math::vec3f wHorizzontal = math::transformVector3F(finalTransform, horizzontal);
			//math::vec3f wPos = math::transformPoint3F(rootM, pos);


			std::shared_ptr<graph::Camera> camera = ops::createNode<graph::Camera>();

			//math::affine3f cameraTransform(wHorizzontal, wUp, -wLookAt, wPos);
			const auto cameraTransform = convertAssimpMatrix(node->mTransformation, swap);
			camera->transform->setAffine(convertAssimpMatrix(node->mTransformation, swap));
			//camera->transform->rotateAroundPoint(camera->transform->affineTransform.p, math::zAxis, M_PI);
			camera->transform->rotateAroundPoint(camera->transform->affineTransform.p, math::yAxis, M_PI_2);
			camera->fovY = fov * 180.0f / M_PI;
			//camera->aspect = aspect;
			camera->updateDirections();

			cameras.push_back(camera);
		}
		return cameras;
	}

	std::tuple<std::shared_ptr<graph::Group>, std::vector<std::shared_ptr<graph::Camera>>> importSceneFile(std::string filePath)
	{
		filePath                     = utl::absolutePath(filePath);
		const std::string fileFormat = utl::getFileExtension(filePath);
		VTX_INFO("Loading scene file: {}", filePath);
		Assimp::Importer importer;
		importer.SetPropertyFloat("PP_GSN_MAX_SMOOTHING_ANGLE", 30);
		const aiScene* scene = importer.ReadFile(filePath,
												 aiProcess_Triangulate |
												 //aiProcess_MakeLeftHanded |
												 //aiProcess_JoinIdenticalVertices |
												 aiProcess_SortByPType |
												 //aiProcess_GenNormals |
												 aiProcess_GenSmoothNormals |
												 aiProcess_CalcTangentSpace |
												 //aiProcess_CalcTangentSpace |
												 //aiProcess_RemoveComponent (remove colors) |
												 //aiProcess_LimitBoneWeights |
												 aiProcess_ImproveCacheLocality |
												 //aiProcess_RemoveRedundantMaterials |
												 //aiProcess_GenUVCoords |
												 aiProcess_FindDegenerates |
												 aiProcess_FindInvalidData |
												 aiProcess_FindInstances |
												 aiTextureFlags_UseAlpha |
												 //aiProcess_ValidateDataStructure |
												 //aiProcess_OptimizeMeshes |
												 //aiProcess_OptimizeGraph |
												 //aiProcess_Debone |
												 0);
		const bool successCondition = (scene && !(scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) && scene->mRootNode);
		VTX_ASSERT_CONTINUE(successCondition, "Assimp Importer Errror: {}", importer.GetErrorString());

		processMetadata(scene, fileFormat);
		std::map<unsigned, vtxID> meshMap;
		VTX_INFO("Creating Scene Graph");
		const std::vector<std::shared_ptr<graph::Material>> importedMaterials = processMaterials(scene, utl::getFolder(filePath));
		std::shared_ptr<graph::Group>                       sceneGraph        = nullptr;
		const std::shared_ptr<graph::Node>                  root              = processAssimpNode(scene->mRootNode, scene, meshMap, importedMaterials, SwapType::yToZ);
		if (root->as<graph::Instance>())
		{
			// scene contains only one mesh, so we need to create a group to contain it
			sceneGraph = ops::createNode<graph::Group>();
			sceneGraph->addChild(root);
		}
		else
		{
			sceneGraph = root->as<graph::Group>();
		}

		sceneGraph->name = "Scene Root Group";

		std::vector<std::shared_ptr<graph::Camera>> cameras = processCameras(scene, SwapType::yToZ);
		return {sceneGraph, cameras};
	}
}

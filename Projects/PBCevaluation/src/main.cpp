#include "../include/PBCevaluation.h"
#include "../include/App.h"

int main()
{

	PBCevaluation pbcevaluation;
	std::string mesh_info = "../../../Projects/PBCevaluation/data/";
	pbcevaluation.tag_file = mesh_info+"meta_periodic_face_tags.csv";
	pbcevaluation.initializeVisualizationMesh(mesh_info+"pbc_visual.obj");
	std::string undeformed_mesh = mesh_info+"pbc_undeformed.obj";
	pbcevaluation.initializeMeshInformation(undeformed_mesh, mesh_info+"exp1/");
	App<PBCevaluation> app(pbcevaluation);
	app.initializeScene();
	app.run();

	return 0;
}
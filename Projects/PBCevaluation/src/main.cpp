#include "../include/PBCevaluation.h"
#include "../include/App.h"

int main()
{

	PBCevaluation pbcevaluation;
	std::string mesh_info = "../../../Projects/PBCevaluation/data/Material2/";
	pbcevaluation.initializeFromDir(mesh_info);
	App<PBCevaluation> app(pbcevaluation);
	app.initializeScene();
	app.run();

	return 0;
}
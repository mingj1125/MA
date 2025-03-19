#include <igl/readOBJ.h>
#include "../include/QuadraticTriangle.h"
#include "../include/App.h"
#include "../include/StiffnessTensor.h"

int main()
{
    QuadraticTriangle quad_tri;

    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid.obj");
    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/sun_mesh_line.obj");
    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid_double_refined.obj");
    // quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/grid_refined_mesh.obj");
    quad_tri.tag_file = "../../../Projects/QuadraticTriangle/data/meta_periodic_face_tags.csv";
    quad_tri.tags = true;
    quad_tri.initializeFromFile("../../../Projects/QuadraticTriangle/data/pbc_deformed.obj");

    // quad_tri.testIsotropicStretch();
    // quad_tri.testVerticalDirectionStretch();
    // quad_tri.testHorizontalDirectionStretch();

    // int static_solve_step = 0;
    // bool finished = quad_tri.advanceOneStep(static_solve_step++);
    // while(!finished) finished = quad_tri.advanceOneStep(static_solve_step++);

    
    App<QuadraticTriangle> app(quad_tri);

    app.initializeScene();
    app.run();

    // std::cout << findStiffnessTensor("../../../Projects/QuadraticTriangle/data/sun_mesh_line.obj")[0] << std::endl;


    return 0;
}

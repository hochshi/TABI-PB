#include "boundary_element.h"

int BoundaryElement::gmres_cpu_(const GmresView& view)
{
    return gmres_impl_(view, false);
}

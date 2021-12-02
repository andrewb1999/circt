//===- MSFTModule.cpp - MSFT API pybind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/MSFT.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"
#include "circt/Support/LLVM.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "PybindUtils.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace circt::msft;
using namespace mlir::python::adaptors;

namespace pybind11 {
namespace detail {
/// Casts object <-> PhysLocationAttr
template <>
struct type_caster<PhysLocationAttr> {
  PYBIND11_TYPE_CASTER(PhysLocationAttr, _("PhysLocationAttr"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    Attribute attr = unwrap(mlirPythonCapsuleToAttribute(capsule.ptr()));
    if (auto physLoc = attr.dyn_cast_or_null<PhysLocationAttr>()) {
      value = physLoc;
      return true;
    }
    return true;
  }
  static handle cast(PhysLocationAttr v, return_value_policy, handle) {
    py::object capsule = py::reinterpret_steal<py::object>(
        mlirPythonAttributeToCapsule(wrap(v)));

    auto attr = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                    .attr("Attribute")
                    .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
                    .release();

    return py::module::import("circt.dialects.msft")
        .attr("PhysLocationAttr")(attr)
        .release();
  }
};
} // namespace detail
} // namespace pybind11

class PrimitiveDB {
public:
  PrimitiveDB(MlirContext ctxt) { db = circtMSFTCreatePrimitiveDB(ctxt); }
  ~PrimitiveDB() { circtMSFTDeletePrimitiveDB(db); }
  bool addPrimitive(MlirAttribute locAndPrim) {
    return mlirLogicalResultIsSuccess(
        circtMSFTPrimitiveDBAddPrimitive(db, locAndPrim));
  }
  bool isValidLocation(MlirAttribute loc) {
    return circtMSFTPrimitiveDBIsValidLocation(db, loc);
  }

  CirctMSFTPrimitiveDB db;
};

class PlacementDB {
public:
  PlacementDB(MlirOperation top, PrimitiveDB *seed) {
    db = circtMSFTCreatePlacementDB(top, seed ? seed->db
                                              : CirctMSFTPrimitiveDB{nullptr});
  }
  ~PlacementDB() { circtMSFTDeletePlacementDB(db); }
  size_t addDesignPlacements() {
    return circtMSFTPlacementDBAddDesignPlacements(db);
  }
  bool addPlacement(MlirAttribute loc, MlirAttribute path, std::string subpath,
                    MlirOperation op) {
    return mlirLogicalResultIsSuccess(circtMSFTPlacementDBAddPlacement(
        db, loc,
        CirctMSFTPlacedInstance{path, subpath.c_str(), subpath.size(), op}));
  }
  py::object getInstanceAt(MlirAttribute loc) {
    CirctMSFTPlacedInstance inst;
    if (!circtMSFTPlacementDBTryGetInstanceAt(db, loc, &inst))
      return py::none();
    std::string subpath(inst.subpath, inst.subpathLength);
    return (py::tuple)py::cast(std::make_tuple(inst.path, subpath, inst.op));
  }
  PhysLocationAttr getNearestFreeInColumn(CirctMSFTPrimitiveType prim,
                                          uint64_t column,
                                          uint64_t nearestToY) {
    MlirAttribute nearest = circtMSFTPlacementDBGetNearestFreeInColumn(
        db, prim, column, nearestToY);
    if (!nearest.ptr)
      return {};
    return unwrap(nearest).cast<PhysLocationAttr>();
  }
  void walkPlacements(
      py::function pycb,
      std::tuple<py::object, py::object, py::object, py::object> bounds,
      py::object prim) {

    auto handleNone = [](py::object o) {
      return o.is_none() ? -1 : o.cast<int64_t>();
    };
    int64_t cBounds[4] = {
        handleNone(std::get<0>(bounds)), handleNone(std::get<1>(bounds)),
        handleNone(std::get<2>(bounds)), handleNone(std::get<3>(bounds))};
    CirctMSFTPrimitiveType cPrim;
    if (prim.is_none())
      cPrim = -1;
    else
      cPrim = prim.cast<CirctMSFTPrimitiveType>();
    circtMSFTPlacementDBWalkPlacements(
        db,
        [](MlirAttribute loc, CirctMSFTPlacedInstance p, void *userData) {
          std::string subpath(p.subpath, p.subpathLength);
          py::gil_scoped_acquire gil;
          py::function pycb = *((py::function *)(userData));
          auto physLoc = unwrap(loc).cast<PhysLocationAttr>();
          if (!p.op.ptr) {
            pycb(physLoc, py::none());
          } else {
            pycb(physLoc, std::make_tuple(p.path, subpath, p.op));
          }
        },
        cBounds, cPrim, &pycb);
  }

private:
  CirctMSFTPlacementDB db;
};

/// Populate the msft python module.
void circt::python::populateDialectMSFTSubmodule(py::module &m) {
  mlirMSFTRegisterPasses();

  m.doc() = "MSFT dialect Python native extension";

  m.def("get_instance", circtMSFTGetInstance, py::arg("root"), py::arg("path"));

  py::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("M20K", PrimitiveType::M20K)
      .value("DSP", PrimitiveType::DSP)
      .export_values();

  mlir_attribute_subclass(m, "PhysLocationAttr",
                          circtMSFTAttributeIsAPhysLocationAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, PrimitiveType devType, uint64_t x, uint64_t y,
             uint64_t num, MlirContext ctxt) {
            return cls(circtMSFTPhysLocationAttrGet(ctxt, (uint64_t)devType, x,
                                                    y, num));
          },
          "Create a physical location attribute", py::arg(),
          py::arg("dev_type"), py::arg("x"), py::arg("y"), py::arg("num"),
          py::arg("ctxt") = py::none())
      .def_property_readonly(
          "devtype",
          [](MlirAttribute self) {
            return (PrimitiveType)circtMSFTPhysLocationAttrGetPrimitiveType(
                self);
          })
      .def_property_readonly("x",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetX(self);
                             })
      .def_property_readonly("y",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetY(self);
                             })
      .def_property_readonly("num", [](MlirAttribute self) {
        return circtMSFTPhysLocationAttrGetNum(self);
      });

  mlir_attribute_subclass(m, "RootedInstancePathAttr",
                          circtMSFTAttributeIsARootedInstancePathAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute rootSymbol,
             std::vector<MlirAttribute> instancePath, MlirContext ctxt) {
            return cls(circtMSFTRootedInstancePathAttrGet(
                ctxt, rootSymbol, instancePath.data(), instancePath.size()));
          },
          "Create an rooted instance path attribute", py::arg(),
          py::arg("root_symbol"), py::arg("instance_path"),
          py::arg("ctxt") = py::none());

  mlir_attribute_subclass(m, "SwitchInstanceAttr",
                          circtMSFTAttributeIsASwitchInstanceAttribute)
      .def_classmethod(
          "get",
          [](py::object cls,
             std::vector<std::pair<MlirAttribute, MlirAttribute>> listOfCases,
             MlirContext ctxt) {
            std::vector<CirctMSFTSwitchInstanceCase> cases;
            for (auto p : listOfCases)
              cases.push_back({std::get<0>(p), std::get<1>(p)});
            return cls(circtMSFTSwitchInstanceAttrGet(ctxt, cases.data(),
                                                      cases.size()));
          },
          "Create an instance switch attribute", py::arg(),
          py::arg("list_of_cases"), py::arg("ctxt") = py::none())
      .def_property_readonly(
          "cases",
          [](MlirAttribute self) {
            size_t numCases = circtMSFTSwitchInstanceAttrGetNumCases(self);
            std::vector<CirctMSFTSwitchInstanceCase> cases(numCases);
            circtMSFTSwitchInstanceAttrGetCases(self, cases.data(),
                                                cases.max_size());
            std::vector<std::pair<MlirAttribute, MlirAttribute>> pyCases;
            for (auto c : cases)
              pyCases.push_back(std::make_pair(c.instance, c.attr));
            return pyCases;
          })
      .def_property_readonly("num_cases", [](MlirAttribute self) {
        return circtMSFTSwitchInstanceAttrGetNumCases(self);
      });

  mlir_attribute_subclass(m, "PhysicalBoundsAttr",
                          circtMSFTAttributeIsAPhysicalBoundsAttr)
      .def_classmethod(
          "get",
          [](py::object cls, uint64_t xMin, uint64_t xMax, uint64_t yMin,
             uint64_t yMax, MlirContext ctxt) {
            auto physicalBounds =
                circtMSFTPhysicalBoundsAttrGet(ctxt, xMin, xMax, yMin, yMax);
            return cls(physicalBounds);
          },
          "Create a PhysicalBounds attribute", py::arg("cls"), py::arg("xMin"),
          py::arg("xMax"), py::arg("yMin"), py::arg("yMax"),
          py::arg("context") = py::none());

  mlir_attribute_subclass(m, "PhysicalRegionRefAttr",
                          circtMSFTAttributeIsAPhysicalRegionRefAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::string name, MlirContext ctxt) {
            auto physicalBounds = circtMSFTPhysicalRegionRefAttrGet(
                ctxt, mlirStringRefCreateFromCString(name.c_str()));
            return cls(physicalBounds);
          },
          "Create a PhysicalRegionRef attribute", py::arg("cls"),
          py::arg("name"), py::arg("context") = py::none());

  py::class_<PrimitiveDB>(m, "PrimitiveDB")
      .def(py::init<MlirContext>(), py::arg("ctxt") = py::none())
      .def("add_primitive", &PrimitiveDB::addPrimitive,
           "Inform the DB about a new placement.", py::arg("loc_and_prim"))
      .def("is_valid_location", &PrimitiveDB::isValidLocation,
           "Query the DB as to whether or not a primitive exists.",
           py::arg("loc"));

  py::class_<PlacementDB>(m, "PlacementDB")
      .def(py::init<MlirOperation, PrimitiveDB *>(), py::arg("top"),
           py::arg("seed") = nullptr)
      .def("add_design_placements", &PlacementDB::addDesignPlacements,
           "Add the placements already present in the design.")
      .def("add_placement", &PlacementDB::addPlacement,
           "Inform the DB about a new placement.", py::arg("location"),
           py::arg("path"), py::arg("subpath"), py::arg("op"))
      .def("get_nearest_free_in_column", &PlacementDB::getNearestFreeInColumn,
           "Find the nearest free primitive location in column.",
           py::arg("prim_type"), py::arg("column"), py::arg("nearest_to_y"))
      .def("get_instance_at", &PlacementDB::getInstanceAt,
           "Get the instance at location. Returns None if nothing exists "
           "there. Otherwise, returns (path, subpath, op) of the instance "
           "there.")
      .def("walk_placements", &PlacementDB::walkPlacements,
           "Walk the placements, with possible bounds. Bounds are (xmin, xmax, "
           "ymin, ymax) with 'None' being unbounded.",
           py::arg("callback"),
           py::arg("bounds") =
               std::make_tuple(py::none(), py::none(), py::none(), py::none()),
           py::arg("prim_type") = py::none());
}

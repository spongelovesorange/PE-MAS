from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Component(Base):
    __tablename__ = "components"

    id = Column(Integer, primary_key=True)
    manufacturer = Column(String, nullable=True)
    mpn = Column(String, nullable=True, index=True)
    category = Column(String, nullable=True, index=True)
    description = Column(Text, nullable=True)
    lifecycle_status = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    offers = relationship("Offer", back_populates="component")
    electrical_parameters = relationship("ElectricalParameter", back_populates="component")
    thermal_parameters = relationship("ThermalParameter", back_populates="component")
    mechanical_data = relationship("MechanicalPackageData", back_populates="component")
    datasheets = relationship("DatasheetLink", back_populates="component")
    simulation_models = relationship("SimulationModelLink", back_populates="component")
    provenance = relationship("SourceProvenance", back_populates="component")


class Distributor(Base):
    __tablename__ = "distributors"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    website = Column(String, nullable=True)
    api_name = Column(String, nullable=True)
    terms_note = Column(Text, nullable=True)

    offers = relationship("Offer", back_populates="distributor")


class Offer(Base):
    __tablename__ = "offers_prices"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    distributor_id = Column(Integer, ForeignKey("distributors.id"), nullable=False)
    sku = Column(String, nullable=True)
    stock_quantity = Column(Integer, nullable=True)
    currency = Column(String, nullable=True)
    unit_price = Column(Float, nullable=True)
    price_break_json = Column(Text, nullable=True)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    component = relationship("Component", back_populates="offers")
    distributor = relationship("Distributor", back_populates="offers")


class ElectricalParameter(Base):
    __tablename__ = "electrical_parameters"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    condition = Column(Text, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    raw_value = Column(String, nullable=True)

    component = relationship("Component", back_populates="electrical_parameters")


class ThermalParameter(Base):
    __tablename__ = "thermal_parameters"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    condition = Column(Text, nullable=True)
    raw_value = Column(String, nullable=True)

    component = relationship("Component", back_populates="thermal_parameters")


class MechanicalPackageData(Base):
    __tablename__ = "package_mechanical_data"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    package_name = Column(String, nullable=True)
    mounting_type = Column(String, nullable=True)
    height_mm = Column(Float, nullable=True)
    length_mm = Column(Float, nullable=True)
    width_mm = Column(Float, nullable=True)
    footprint = Column(String, nullable=True)
    raw_package = Column(Text, nullable=True)

    component = relationship("Component", back_populates="mechanical_data")


class DatasheetLink(Base):
    __tablename__ = "datasheet_links"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(String, nullable=True)
    retrieved = Column(Boolean, default=False)
    license_note = Column(Text, nullable=True)

    component = relationship("Component", back_populates="datasheets")


class SimulationModelLink(Base):
    __tablename__ = "simulation_model_links"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=False)
    model_type = Column(String, nullable=True)
    url_or_path = Column(Text, nullable=True)
    status = Column(String, nullable=True)
    license_note = Column(Text, nullable=True)

    component = relationship("Component", back_populates="simulation_models")


class SourceProvenance(Base):
    __tablename__ = "source_provenance"

    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey("components.id"), nullable=True)
    source = Column(String, nullable=False)
    url_or_api = Column(Text, nullable=True)
    retrieval_date = Column(String, nullable=True)
    license_or_usage_note = Column(Text, nullable=True)
    confidence = Column(String, nullable=True)
    raw_payload_hash = Column(String, nullable=True)

    component = relationship("Component", back_populates="provenance")

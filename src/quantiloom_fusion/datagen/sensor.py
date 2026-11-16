"""Per-band sensor configuration TOML content."""

import math

from ..core.hashing import fingerprint


def sensor_config(band,c,d,identity):
    w,h=c["resolution"]
    focal=(h*c["pixel_pitch_um"]*1e-3)/(2*math.tan(math.radians(c["fov_y_deg"])/2))
    # Three policies. per_acquisition keeps one detector seed per band -- the
    # FPN classes ignore the acquisition counter by shader design, so PRNU/DSNU
    # stay put across views -- while dataset.acquisition_index shifts the
    # counter the temporal classes key on, so each capture gets its own
    # temporal realisation. per_sample reseeds the detector per capture (all
    # classes change, FPN included). fixed_detector is the legacy stream.
    key=identity if d["noise_seed_policy"]=="per_sample" else "fixed_detector"
    sensor={"version":1,"enabled":True,"input_kind":"spectral",
        "noise_seed":int(fingerprint([d['seed'],band,key])[:8],16),
        "preset":{"vis":"generic_cmos","mwir":"generic_cooled_photon_ir","lwir":"generic_uncooled_thermal_ir"}[band],
        "optics":{"focal_length_mm":focal,"pixel_pitch_um":c["pixel_pitch_um"],"f_number":c["f_number"],"sensor_width_px":w,"sensor_height_px":h,"psf_sigma_px":c["psf_sigma_px"],
                  "projection":{"model":"brown_conrady","coefficients":c["distortion"]}},
        "exposure":{"time_s":c["exposure_s"]},
        "readout":{"shutter":"global","frame_period_s":max(1/30,c['exposure_s']),"analog_gain":1.0,"adc_bits":c["adc_bits"],"output_bits":c["adc_bits"]},
        "quality":{"noise_free":c["noise_free"],"wavelength_samples":d["wavelength_samples"],"time_samples":1,"pixel_samples":1},
        "products":{"traced_radiance":False,"cie_linear_srgb":False,"band_measurement":True,"raw_dn":True,"corrected_device_signal":True,"apparent_temperature":band!="vis","display":True},
        "warmup":{"seconds":c["warmup_s"]}}
    if band=="lwir":
        sensor.update(effective_min_nm=8000.0,effective_max_nm=12000.0)
        sensor["channels"]=[{"name":"Mono","absorptance":{"kind":"thermal_absorptance","wavelength_nm":[8000.0,12000.0],"value":[.8,.8]}}]
        # The whole fixed pattern is stated here rather than inherited from the
        # preset, so the dataset does not move when the preset's defaults do.
        sensor["thermal"]={"netd_k":c["netd_k"],"responsivity_dn_w":c["responsivity_dn_w"],"time_constant_s":c["time_constant_s"],
            "enable_fpn":True,**{k:c[k] for k in ("fpn_gain_sigma","fpn_offset_k","fpn_column_gain_sigma","fpn_column_offset_k","fpn_row_gain_sigma","fpn_row_offset_k")},
            "apply_nuc":c["apply_nuc"],"nuc_residual_fraction":c["nuc_residual_fraction"]}
    else:
        sensor["photon"]={"full_well_e":c["full_well_e"],"electrons_per_dn":c["electrons_per_dn"],"read_noise_e_rms":c["read_noise_e_rms"],"prnu_sigma":c["prnu_sigma"],"dsnu_e_rms":c["dsnu_e_rms"],"enable_fpn":True,
            **{k:c[k] for k in ("column_gain_sigma","column_offset_e_rms","row_gain_sigma","row_offset_e_rms")},
            "apply_nuc":c["apply_nuc"],"nuc_residual_fraction":c["nuc_residual_fraction"]}
        # electrons_per_dn is a readout field, never a photon parameter.
        sensor["readout"]["electrons_per_dn"]=sensor["photon"].pop("electrons_per_dn")
        # Optional; absent keys leave the preset's value and the job unchanged.
        # Written as floats: the renderer reads both with GetDouble.
        if "black_level_dn" in c:sensor["readout"]["black_level_dn"]=float(c["black_level_dn"])
        if "bias_dn_rms" in c:sensor["photon"]["bias_dn_rms"]=float(c["bias_dn_rms"])
        if band=="vis":
            sensor["channels"]=[]
            for name,wave,val in [("R",[400.,540.,610.,700.],[0.,0.,.55,.25]),("G",[400.,470.,540.,620.,700.],[0.,.2,.6,.1,0.]),("B",[400.,460.,520.,700.],[.2,.55,0.,0.])]:
                sensor["channels"].append({"name":name,"qe":{"kind":"absolute_qe","wavelength_nm":wave,"value":val}})
    return sensor

import bilby
import numpy as np
from bilby.core.utils import ra_dec_to_theta_phi, speed_of_light
from bilby.core.utils.constants import *
import bilby.gw.utils as gwutils
from copy import deepcopy
import optparse
import os
import json
import pickle
import hashlib

# parse commands
parser = optparse.OptionParser()
parser.add_option("--chirp_mass", type="float")
parser.add_option("--chirp_mass_sigma", type="float")
parser.add_option("--mass_ratio", dest="mass_ratio", type="float")
parser.add_option("--mass_ratio_sigma", dest="mass_ratio_sigma", type="float")
parser.add_option("--luminosity_distance", dest="luminosity_distance", type="float")
parser.add_option("--luminosity_distance_sigma", dest="luminosity_distance_sigma", type="float")
parser.add_option("--chi_1", dest="chi_1", type="float")
parser.add_option("--chi_1_sigma", dest="chi_1_sigma", type="float")
parser.add_option("--chi_2", dest="chi_2", type="float")
parser.add_option("--chi_2_sigma", dest="chi_2_sigma", type="float")
parser.add_option("--ra", dest="ra", type="float")
parser.add_option("--dec", dest="dec", type="float")
parser.add_option("--theta_jn", dest="theta_jn", type="float")
parser.add_option("--phase", dest="phase", type="float")
parser.add_option("--geocent_time", dest="geocent_time", type="float")
parser.add_option("--geocent_time_sigma", dest="geocent_time_sigma", type="float")
parser.add_option("--psi", dest="psi", type="float")
parser.add_option("--nlive", dest="nlive", type="int", default = 4000)
parser.add_option("--walks", type = "int", default = 100)
parser.add_option("--maxmcmc", type = "int", default = 5000)
parser.add_option("--dlogz", dest="dlogz", type="float", default = 1e-5)
parser.add_option("--a", type="float", default = 0)
parser.add_option("--chi", type="int", default = 20)
parser.add_option("--epsilon", type="float", default = 0.1)
parser.add_option("--randomseed", dest="randomseed", type="int", default=569)
parser.add_option("--outdir", dest="outdir", type="string", default = 'results/')
parser.add_option("--flow", dest="flow", type="float", default=5)
parser.add_option("--srate", dest="srate", type="float", default=4096.)
parser.add_option("--fref", dest="fref", type="float", default=100.)
parser.add_option("--tc_offset", dest="tc_offset", type="float", default=1.)
parser.add_option('-1',"--no_earth_rotation_time_delay_inj", dest="ertd_inj", action="store_false", default=True)
parser.add_option('-4',"--no_earth_rotation_time_delay_ana", dest="ertd_ana", action="store_false", default=True)
parser.add_option('-2',"--no_earth_rotation_beam_patterns_inj", dest="erbp_inj", action="store_false", default=True)
parser.add_option('-5',"--no_earth_rotation_beam_patterns_ana", dest="erbp_ana", action="store_false", default=True)
parser.add_option('-3',"--no_finite_size_inj", dest="fs_inj", action="store_false", default=True)
parser.add_option('-6',"--no_finite_size_ana", dest="fs_ana", action="store_false", default=True)
parser.add_option("--label", dest="label", type="string")
parser.add_option("--sample-CE-arrival-time", action="store_true", default=False, dest="sample_CE_arrival_time")
parser.add_option("--only_extrinsic", action="store_true", default=False, dest="only_extrinsic")
parser.add_option("--mode_array", default='[[2,2]]', type='string' )
parser.add_option("--only_extrinsic_plus_chirp_mass", action="store_true", default=False, dest="only_extrinsic_plus_chirp_mass")
parser.add_option("--A1", action="store_true", default=False)
parser.add_option("--CE20", action="store_true", default=False)
parser.add_option("--waveformname", type = 'string', default = 'IMRPhenomXPHM')
parser.add_option("--nact", dest="nact", type="float", default=5.)
parser.add_option("--mode_array_injection", type='string' )
parser.add_option("--npool", type="int", default=4,
                  help="sampler worker processes; keep equal to condor request_cpus")
parser.add_option("--prior_nsigma", type="float", default=3.,
                  help="half-width of the chirp_mass/mass_ratio/distance prior boxes, in units of the supplied Fisher sigmas")
parser.add_option("--time_prior_halfwidth", type="float", default=0.001,
                  help="half-width [s] of the arrival-time prior; posteriors use <0.12 ms and corr(A, t)~0, so 1 ms is >9 sigma for the worst event")
parser.add_option("--asd_CE40high", action="store_true", default=False, help="use the 1.5 MW CE40 ASD (default)")
parser.add_option("--asd_CE40low", action="store_true", default=False, help="use the 1.0 MW CE40 ASD")
parser.add_option("--asd_CE20high", action="store_true", default=False, help="use the 1.5 MW CE20 ASD (default)")
parser.add_option("--asd_CE20low", action="store_true", default=False, help="use the 1.0 MW CE20 ASD")
parser.add_option("--asd_A1", type="string", default='/ligo/home/ligo.org/pratyusava.baral/dispersion/asd/Aplus_asd.txt')



(options, args) = parser.parse_args()

required = ["chirp_mass", "chirp_mass_sigma", "mass_ratio", "mass_ratio_sigma",
            "luminosity_distance", "luminosity_distance_sigma", "chi_1", "chi_2",
            "ra", "dec", "theta_jn", "phase", "geocent_time", "psi"]
missing = [k for k in required if getattr(options, k) is None]
if missing:
    parser.error("missing required options: " + ", ".join("--" + k for k in missing))

# resolve detector sensitivities (default: high = 1.5 MW laser power)
if options.asd_CE40high and options.asd_CE40low:
    parser.error("--asd_CE40high and --asd_CE40low are mutually exclusive")
if options.asd_CE20high and options.asd_CE20low:
    parser.error("--asd_CE20high and --asd_CE20low are mutually exclusive")
ASD_DIR = '/ligo/home/ligo.org/pratyusava.baral/dispersion/asd'
asd_CE40 = f"{ASD_DIR}/CE40km_{'1p0' if options.asd_CE40low else '1p5'}MW_aLIGO_coat_strain.txt"
asd_CE20 = f"{ASD_DIR}/CE20km_{'1p0' if options.asd_CE20low else '1p5'}MW_aLIGO_coat_strain.txt"
print(f"CE40 ASD: {asd_CE40}")

# offset the seed by the event time: reproducible per event, but sampler
# randomness is not shared across the whole injection campaign
np.random.seed((options.randomseed + int(options.geocent_time)) % 2**32)

outdir = options.outdir

os.makedirs(outdir, exist_ok=True)


mode_array = json.loads(str(options.mode_array))
if options.mode_array_injection is None:
    mode_array_injection = mode_array
else:
    mode_array_injection = json.loads(str(options.mode_array_injection))


chirp_mass = options.chirp_mass
mass_ratio = options.mass_ratio
chi_1 = options.chi_1
chi_2 = options.chi_2
ra = options.ra
dec = options.dec
theta_jn = options.theta_jn
psi = options.psi
phase=options.phase
geocent_time=options.geocent_time
luminosity_distance = options.luminosity_distance

a = options.a

injection_parameters = dict(
    chirp_mass=chirp_mass, mass_ratio=mass_ratio, chi_1=chi_1, chi_2=chi_2,
    ra=ra, dec=dec, luminosity_distance = luminosity_distance,
    theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=geocent_time, a=a , A=0, fiducial=1,
)
    
minimum_frequency = options.flow
sampling_frequency = options.srate
reference_frequency = options.fref
tc_offset = options.tc_offset
waveformname = options.waveformname

chirp_mass_in_seconds = chirp_mass * solar_mass * gravitational_constant / speed_of_light**3.
t0 = -5. / 256. * chirp_mass_in_seconds * (np.pi * chirp_mass_in_seconds * minimum_frequency)**(-8. / 3.)
duration = 2**(np.int32(np.log2(np.abs(t0)))+1)
start_time = injection_parameters["geocent_time"] - duration + tc_offset


waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_individual_modes,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(waveform_approximant=waveformname, reference_frequency=reference_frequency, minimum_frequency=minimum_frequency, mode_array=mode_array_injection))

frequencies_asd, strain_asd = np.loadtxt(asd_CE40, unpack=True)
        
ifos = []
ifo = bilby.gw.detector.load_interferometer('/ligo/home/ligo.org/pratyusava.baral/dispersion/detector_configurations/bilby/CE40.ifo')
ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies_asd,
            asd_array=strain_asd
        )
ifos.append(ifo)

if options.A1:
    frequencies_asd, strain_asd = np.loadtxt(options.asd_A1, unpack=True)
    ifo = bilby.gw.detector.load_interferometer('/ligo/home/ligo.org/pratyusava.baral/dispersion/detector_configurations/bilby/A1.ifo')
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies_asd,
            asd_array=strain_asd
        )
    ifos.append(ifo)

if options.CE20:
    print(f"CE20 ASD: {asd_CE20}")
    frequencies_asd, strain_asd = np.loadtxt(asd_CE20, unpack=True)
    ifo = bilby.gw.detector.load_interferometer('/ligo/home/ligo.org/pratyusava.baral/dispersion/detector_configurations/bilby/CE20.ifo')
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=frequencies_asd,
            asd_array=strain_asd
        )
    ifos.append(ifo)
    

frequencies = waveform_generator.frequency_array
idxs_above_minimum_frequency = frequencies > (minimum_frequency - (frequencies[1] - frequencies[0]))
freqs = frequencies[idxs_above_minimum_frequency]

converted_injection_parameters, _ = waveform_generator.parameter_conversion(injection_parameters)
waveform_polarizations = waveform_generator.frequency_domain_strain(converted_injection_parameters)
waveform_polarizations_reduced = {}

for key in waveform_polarizations.keys():
    waveform_polarizations_reduced[key] = {}
    waveform_polarizations_reduced[key]['plus'] = waveform_polarizations[key]['plus'][idxs_above_minimum_frequency]
    waveform_polarizations_reduced[key]['cross'] = waveform_polarizations[key]['cross'][idxs_above_minimum_frequency]

for ifo in ifos:
    h = np.zeros_like(waveform_generator.frequency_array, dtype=complex)
    h[idxs_above_minimum_frequency] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
        waveform_polarizations = waveform_polarizations_reduced,
        parameters = converted_injection_parameters,
        start_time = start_time,
        frequencies = freqs,
        earth_rotation_time_delay=options.ertd_inj,
        earth_rotation_beam_patterns=options.erbp_inj,
        finite_size=options.fs_inj)
            
    ifo.set_strain_data_from_frequency_domain_strain(
            h, start_time=start_time, frequency_array=frequencies
            )
    ifo.minimum_frequency = minimum_frequency
    snr_squared = ifo.optimal_snr_squared(signal=ifo.frequency_domain_strain)
    snr = np.sqrt(snr_squared)
    print(f"The injected SNR is {snr} in {ifo}.")

# set up priors
# construct priors
priors = bilby.gw.prior.BNSPriorDict()
for key in ["mass_1", "mass_2", "lambda_1", "lambda_2"]:
    priors.pop(key)

# prior boxes at injected value +/- prior_nsigma Fisher sigmas; the Fisher
# sigmas assume a unimodal posterior around the truth, so a degenerate
# (mirror sky/inclination) mode can rail against these edges if too narrow
ns = options.prior_nsigma
priors["chirp_mass"] = bilby.core.prior.Uniform(name='chirp_mass', minimum=chirp_mass - ns*options.chirp_mass_sigma, maximum=chirp_mass + ns*options.chirp_mass_sigma)
priors["mass_ratio"] = bilby.core.prior.Uniform(name='mass_ratio', minimum=max(0.125, mass_ratio - ns*options.mass_ratio_sigma), maximum=min(1,mass_ratio + ns*options.mass_ratio_sigma))
# priors["chi_1"] = bilby.core.prior.Uniform(name='chi_1', minimum=chi_1 - 3*options.chi_1_sigma, maximum=chi_1 + 3*options.chi_1_sigma)
# priors["chi_2"] = bilby.core.prior.Uniform(name='chi_2', minimum=chi_2 - 3*options.chi_2_sigma, maximum=chi_2 + 3*options.chi_2_sigma)
priors["a"] = options.a
# NOTE: the injected value A=0 lies OUTSIDE this prior's support (|A| >= 1e-5).
# Deliberate: this is a null test quoting upper limits, and the sampler does
# not converge with a Uniform prior on A. Consequences: the posterior can
# never concentrate on the truth, and P-P/coverage tests on A are meaningless.
# For a flat-in-A statement, reweight samples by |A| (see combine_posterior.py).
#priors['A'] = bilby.core.prior.Uniform(name='A', minimum=-1, maximum=1)
priors['A'] = bilby.core.prior.SymmetricLogUniform(name='A', minimum=1e-5, maximum=1e-2)

priors["luminosity_distance"] = bilby.core.prior.Uniform(name='luminosity_distance', minimum=max(10, luminosity_distance - ns*options.luminosity_distance_sigma), maximum=luminosity_distance + ns*options.luminosity_distance_sigma)


if options.sample_CE_arrival_time:
    reference_ifo = bilby.gw.detector.get_empty_interferometer("CE")
    injection_parameters["CE_time"] = injection_parameters['geocent_time'] + reference_ifo.time_delay_from_geocenter(
        ra=injection_parameters['ra'], dec=injection_parameters['dec'], time=injection_parameters['geocent_time'])
    #print (freqs[-1], frequencies[-1])
    priors['CE_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters["CE_time"] - options.time_prior_halfwidth,
        maximum=injection_parameters["CE_time"] + options.time_prior_halfwidth,
        name='CE_time', latex_label='$t_{CE}$', unit='$s$'
    )
else:
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=injection_parameters['geocent_time'] - options.time_prior_halfwidth,
        maximum=injection_parameters['geocent_time'] + options.time_prior_halfwidth,
        name='geocent_time', latex_label='$t_c$', unit='$s$'
    )

# set up likelihood
# tie the pickle filename to everything that defines the likelihood, so a
# stale pickle from a different configuration is never silently reused on
# restart (sampler-only settings excluded: they don't change the likelihood)
sampler_only = {"nlive", "walks", "maxmcmc", "dlogz", "nact", "npool",
                "outdir", "randomseed", "label"}
likelihood_config = {k: v for k, v in sorted(options.__dict__.items())
                     if k not in sampler_only}
likelihood_config["injection_parameters"] = injection_parameters
config_hash = hashlib.md5(json.dumps(likelihood_config, sort_keys=True,
                                     default=str).encode()).hexdigest()[:8]
path_to_likelihood = os.path.join(outdir, f"likelihood_{options.label}_{config_hash}.pickle")

# print (0)

if not os.path.exists(path_to_likelihood):
    # print (waveformname)
    search_waveform_generator_RB = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole_relative_binning_individual_modes,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(waveform_approximant=waveformname, reference_frequency=reference_frequency, minimum_frequency=minimum_frequency, mode_array=mode_array))

    priors['fiducial'] = 0    
    
    if options.sample_CE_arrival_time:
        time_reference = "CE"
    else:
        time_reference = "geocent"
        
    

    likelihood_RB = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransientNextGenerationModebyMode(
        interferometers=ifos, waveform_generator=search_waveform_generator_RB, priors=priors,
        distance_marginalization=False,
        phase_marginalization=False, 
        time_reference=time_reference,
        chi=options.chi,
        epsilon=options.epsilon,
        fiducial_parameters=injection_parameters,
        earth_rotation_time_delay = options.ertd_ana,
        earth_rotation_beam_patterns = options.erbp_ana,
        finite_size = options.fs_ana,
    )
    pickle.dump(likelihood_RB, open(path_to_likelihood, "wb"))
else:
    likelihood_RB = pickle.load(open(path_to_likelihood, "rb"))


result = bilby.run_sampler(
    likelihood=likelihood_RB, priors=priors, sampler='dynesty', use_ratio=True,
    nlive=options.nlive, walks=options.walks, maxmcmc=options.maxmcmc, naccept=options.nact, npool=options.npool,
    injection_parameters=injection_parameters, sample = 'acceptance-walk',
    outdir=outdir, label=options.label, dlogz=options.dlogz)#FIXME 
  

# Make a corner plot.
result.plot_corner()


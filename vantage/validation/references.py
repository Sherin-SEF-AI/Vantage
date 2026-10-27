"""The literature, standards and data the checks and the technical reference (site/methods.html) cite, by key."""

REFERENCES = {
    'iso8855': 'ISO 8855:2011, Road vehicles — Vehicle dynamics and road-holding ability — Vocabulary.',
    'rep103': 'ROS REP 103, Standard Units of Measure and Coordinate Conventions (T. Foote, M. Purvis, 2010).',
    'hartley2004': 'R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., Cambridge '
                   'University Press, 2004.',
    'kannala2006': 'J. Kannala and S. S. Brandt, "A generic camera model and calibration method for conventional, '
                   'wide-angle, and fish-eye lenses", IEEE Trans. Pattern Analysis and Machine Intelligence 28(8), '
                   '1335–1340, 2006.',
    'miyamoto1964': 'K. Miyamoto, "Fish eye lens", Journal of the Optical Society of America 54(8), 1060–1061, 1964.',
    'opencv': 'OpenCV, Camera Calibration and 3D Reconstruction (calib3d) and its fisheye module (cv::fisheye), '
              'https://docs.opencv.org/.',
    'econ': 'e-con Systems, product pages and lens datasheets of the camera modules in the built-in library '
            '(each lens names its page), https://www.e-consystems.com.',
    'zhang2000': 'Z. Zhang, "A flexible new technique for camera calibration", IEEE Trans. Pattern Analysis and '
                 'Machine Intelligence 22(11), 1330–1334, 2000.',
    'greene1986': 'N. Greene, "Environment mapping and other applications of world projections", IEEE Computer '
                  'Graphics and Applications 6(11), 21–29, 1986.',
    'moller1997': 'T. Möller and B. Trumbore, "Fast, minimum storage ray–triangle intersection", Journal of Graphics '
                  'Tools 2(1), 21–28, 1997.',
    'johnson1958': 'J. Johnson, "Analysis of image forming systems", Image Intensifier Symposium, Fort Belvoir, 1958 '
                   '(the pixels-on-target criteria for detection and recognition).',
    'skolnik2001': 'M. I. Skolnik, Introduction to Radar Systems, 3rd ed., McGraw-Hill, 2001 (the radar equation).',
    'knott2004': 'E. F. Knott, J. F. Shaeffer and M. T. Tuley, Radar Cross Section, 2nd ed., SciTech, 2004 '
                 '(the trihedral corner reflector, σ = 4πa⁴ / 3λ²).',
    'macleod2010': 'H. A. Macleod, Thin-Film Optical Filters, 4th ed., CRC Press, 2010, ch. 2 (the characteristic '
                   'matrix of a multilayer).',
    'bornwolf1999': 'M. Born and E. Wolf, Principles of Optics, 7th ed., Cambridge University Press, 1999, §1.5–1.6 '
                    '(Fresnel equations, stratified media).',
    'emilsson2017': 'E. Emilsson, Radar Transparency and Paint Compatibility: A Study of Automobile Bumper and '
                    'Bumper-Skin Complex Permittivities for 77 GHz Microwaves, master\'s thesis, Chalmers University of '
                    'Technology, 2017 (sample 11: measured S21 and S11 of a painted plate).',
    'itu_p2040': 'Recommendation ITU-R P.2040-3, Effects of building materials and structures on radiowave '
                 'propagation above about 100 MHz, 2023.',
    'hecht2017': 'E. Hecht, Optics, 5th ed., Pearson, 2017 (the lateral shift of a ray through a plane-parallel '
                 'plate).',
    'koschmieder1924': 'H. Koschmieder, "Theorie der horizontalen Sichtweite", Beiträge zur Physik der freien '
                       'Atmosphäre 12, 33–55 and 171–181, 1924.',
    'wmo8': 'WMO-No. 8, Guide to Instruments and Methods of Observation, World Meteorological Organization '
            '(meteorological optical range: the distance at which the contrast of a black object falls to 5 %).',
    'itu_p1814': 'Recommendation ITU-R P.1814, Prediction methods required for the design of terrestrial free-space '
                 'optical links, 2007 (rain and snow attenuation).',
    'kim2001': 'I. I. Kim, B. McArthur and E. Korevaar, "Comparison of laser beam propagation at 785 nm and 1550 nm '
               'in fog and haze for optical wireless communications", Proc. SPIE 4214, 26–37, 2001.',
    'itu_p838': 'Recommendation ITU-R P.838-3, Specific attenuation model for rain for use in prediction methods, '
                '2005.',
    'itu_p840': 'Recommendation ITU-R P.840, Attenuation due to clouds and fog (the double-Debye permittivity of '
                'liquid water and the Rayleigh absorption coefficient K_l).',
    'kunkel1984': 'B. A. Kunkel, "Parameterization of droplet terminal velocity and extinction coefficient in fog '
                  'models", Journal of Climate and Applied Meteorology 23, 34–41, 1984.',
    'corless1996': 'R. M. Corless, G. H. Gonnet, D. E. G. Hare, D. J. Jeffrey and D. E. Knuth, "On the Lambert W '
                   'function", Advances in Computational Mathematics 5, 329–359, 1996.',
    'unr112': 'UN Regulation No. 112, Uniform provisions concerning the approval of motor vehicle headlamps emitting '
              'an asymmetrical passing beam or a driving beam or both (class B photometric test points).',
    'janesick2007': 'J. R. Janesick, Photon Transfer, SPIE Press, 2007 (signal, shot noise and read noise of an '
                    'image sensor).',
    'reda2004': 'I. Reda and A. Andreas, "Solar position algorithm for solar radiation applications", Solar Energy '
                '76(5), 577–589, 2004 (NREL SPA; its worked example).',
    'meeus1998': 'J. Meeus, Astronomical Algorithms, 2nd ed., Willmann-Bell, 1998 (ch. 25 solar coordinates, ch. 28 '
                 'the equation of time).',
    'noaa': 'NOAA Global Monitoring Laboratory, General Solar Position Calculations and the NOAA Solar Calculator.',
    'gillespie1992': 'T. D. Gillespie, Fundamentals of Vehicle Dynamics, SAE International, 1992 (ride rates, load '
                     'transfer, anti-dive and anti-squat, roll centres).',
    'milliken1995': 'W. F. Milliken and D. L. Milliken, Race Car Vehicle Dynamics, SAE International, 1995 (wheel '
                    'rate, motion ratio, roll stiffness and lateral load transfer).',
    'aashto2018': 'AASHTO, A Policy on Geometric Design of Highways and Streets (the Green Book), 7th ed., 2018 '
                  '(crest and sag vertical curves).',
    'liang2008': 'C.-K. Liang, L.-W. Chang and H. H. Chen, "Analysis and compensation of rolling shutter effect", '
                 'IEEE Trans. Image Processing 17(8), 1323–1330, 2008.',
    'ieee1588': 'IEEE 1588-2019, Standard for a Precision Clock Synchronization Protocol for Networked Measurement '
                'and Control Systems (PTP).',
    'groves2013': 'P. D. Groves, Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd ed., '
                  'Artech House, 2013 (lever arms, attitude error propagation).',
    'triggs2000': 'B. Triggs, P. F. McLauchlan, R. I. Hartley and A. W. Fitzgibbon, "Bundle adjustment — a modern '
                  'synthesis", Vision Algorithms: Theory and Practice, LNCS 1883, 298–372, 2000.',
    'huber1964': 'P. J. Huber, "Robust estimation of a location parameter", Annals of Mathematical Statistics 35(1), '
                 '73–101, 1964.',
    'huber1967': 'P. J. Huber, "The behavior of maximum likelihood estimates under nonstandard conditions", Proc. '
                 'Fifth Berkeley Symposium on Mathematical Statistics and Probability 1, 221–233, 1967 (the sandwich '
                 'covariance).',
    'kay1993': 'S. M. Kay, Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory, Prentice Hall, '
               '1993 (Fisher information and the Cramér–Rao bound).',
    'zhang2004': 'Q. Zhang and R. Pless, "Extrinsic calibration of a camera and laser range finder (improves camera '
                 'calibration)", Proc. IEEE/RSJ IROS, 2301–2306, 2004.',
    'unnikrishnan2005': 'R. Unnikrishnan and M. Hebert, Fast extrinsic calibration of a laser rangefinder to a '
                        'camera, Carnegie Mellon University, CMU-RI-TR-05-09, 2005.',
    'nuscenes2020': 'H. Caesar et al., "nuScenes: A multimodal dataset for autonomous driving", Proc. IEEE/CVF CVPR, '
                    '11621–11631, 2020; the nuScenes devkit and its schema.',
    'kitti2013': 'A. Geiger, P. Lenz, C. Stiller and R. Urtasun, "Vision meets robotics: The KITTI dataset", '
                 'International Journal of Robotics Research 32(11), 1231–1237, 2013; the KITTI raw data devkit.',
    'kalibr2013': 'P. Furgale, J. Rehder and R. Siegwart, "Unified temporal and spatial calibration for multi-sensor '
                  'systems", Proc. IEEE/RSJ IROS, 1280–1286, 2013; Kalibr\'s camchain.yaml format.',
    'ros_camera_info': 'ROS sensor_msgs/CameraInfo message definition and the camera_calibration_parsers YAML '
                       'format.',
    'saltelli2010': 'A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "Variance based '
                    'sensitivity analysis of model output. Design and estimator for the total sensitivity index", '
                    'Computer Physics Communications 181(2), 259–270, 2010.',
    'sobol2001': 'I. M. Sobol\', "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo '
                 'estimates", Mathematics and Computers in Simulation 55(1–3), 271–280, 2001.',
    'saltelli2002': 'A. Saltelli, "Making best use of model evaluations to compute sensitivity indices", Computer '
                    'Physics Communications 145(2), 280–297, 2002.',
    'jansen1999': 'M. J. W. Jansen, "Analysis of variance designs for model output", Computer Physics Communications '
                  '117(1–2), 35–43, 1999.',
    'ishigami1990': 'T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for '
                    'computer models", Proc. ISUMA \'90, 398–403, 1990.',
    'morris1991': 'M. D. Morris, "Factorial sampling plans for preliminary computational experiments", Technometrics '
                  '33(2), 161–174, 1991.',
    'campolongo2007': 'F. Campolongo, J. Cariboni and A. Saltelli, "An effective screening design for sensitivity '
                      'analysis of large models", Environmental Modelling & Software 22(10), 1509–1518, 2007.',
    'mckay1979': 'M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values '
                 'of input variables in the analysis of output from a computer code", Technometrics 21(2), 239–245, '
                 '1979.',
    'efron1993': 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.',
    'wilson1927': 'E. B. Wilson, "Probable inference, the law of succession, and statistical inference", Journal of '
                  'the American Statistical Association 22(158), 209–212, 1927.',
}


def cite(keys) -> list:
    """[(key, text)] for the keys known (unknown ones are left out)."""
    return [(k, REFERENCES[k]) for k in keys if k in REFERENCES]

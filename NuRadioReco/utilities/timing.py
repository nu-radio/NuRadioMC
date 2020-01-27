import logging
logger = logging.getLogger('NuRadioReco.timing')


def analyze_timing(module_list, t_tot=None):
    data = []
    t_tot_int = 0
    for module in module_list:
        dt = module.end()
        data.append([module.__class__.__name__, dt])
        t_tot_int += dt.total_seconds()
    logger.info("timing information")
    for name, dt in data:
        logger.info("{:<30} \t{}\t{:>6.1f}%\t{:>6.1f}%".format(name, dt, 100. * dt.total_seconds() / t_tot_int,
                                                 100. * dt.total_seconds() / t_tot))

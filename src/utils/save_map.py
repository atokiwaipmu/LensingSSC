
import bigfile

def save_map(cat, fname, zs, kappa, Nm, kappabar, zlmin, zlmax, ds, nbar, nside):
    # save the maps
    cat.logger.info("saving maps to %s" % fname)
    cat.logger.info("started writing source plane %s" % zs)

    with bigfile.File(fname, create=True) as ff:
        ds1 = ff.create_from_array("kappa", kappa, Nfile=1)
        ds2 = ff.create_from_array("Nm", Nm, Nfile=1)

        for d in ds1, ds2:
            d.attrs['kappabar'] = kappabar
            d.attrs['nside'] = nside
            d.attrs['zlmin'] = zlmin
            d.attrs['zlmax'] = zlmax
            d.attrs['zs'] = zs
            d.attrs['ds'] = ds
            d.attrs['nbar'] = nbar

        cat.logger.info("source plane %s written" % zs)
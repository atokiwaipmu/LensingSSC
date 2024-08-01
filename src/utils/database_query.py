import sqlite3

def query_fits_paths(db_path='/lustre/work/akira.tokiwa/Projects/LensingSSC/results/kappa_datapath.db', config_sim=None, zs=None, sl=None, survey=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    query = 'SELECT file_path FROM FitsData WHERE 1=1'
    params = []

    if config_sim is not None:
        query += ' AND config_sim = ?'
        params.append(config_sim)
    if zs is not None:
        query += ' AND zs = ?'
        params.append(zs)
    if sl is not None:
        query += ' AND sl = ?'
        params.append(sl)
    if survey is not None:
        query += ' AND survey = ?'
        params.append(survey)

    c.execute(query, tuple(params))
    results = c.fetchall()
    conn.close()
    return results
import os
import numpy as np
import sqlite3
import pickle

class DatabaseConstructor:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kappa_maps (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                data BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noisy_maps (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                data BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smoothed_maps (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                sl INTEGER,
                data BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noisy_patches (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                oa REAL,
                data BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smoothed_patches (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                sl REAL,
                oa REAL,
                data BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noisy_map_stats (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                power_spectrum BLOB,
                pdf BLOB,
                peak BLOB,
                minima BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smoothed_map_stats (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                sl REAL,
                power_spectrum BLOB,
                pdf BLOB,
                peak BLOB,
                minima BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS noisy_patch_stats (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                oa REAL,
                bispectrum BLOB,
                power_spectrum BLOB,
                pdf BLOB,
                peak BLOB,
                minima BLOB
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS smoothed_patch_stats (
                id INTEGER PRIMARY KEY,
                box_type TEXT,
                seed INTEGER,
                zs REAL,
                ngal INTEGER,
                sl REAL,
                oa REAL,
                bispectrum BLOB,
                power_spectrum BLOB,
                pdf BLOB,
                peak BLOB,
                minima BLOB
            )
        ''')
        self.conn.commit()

    def insert_kappa_map(self, box_type, seed, zs, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO kappa_maps (box_type, seed, zs, data)
            VALUES (?, ?, ?, ?)
        ''', (box_type, seed, zs, pickle.dumps(data)))
        self.conn.commit()

    def insert_noisy_map(self, box_type, seed, zs, ngal, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO noisy_maps (box_type, seed, zs, ngal, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, pickle.dumps(data)))
        self.conn.commit()

    def insert_smoothed_map(self, box_type, seed, zs, ngal, sl, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO smoothed_maps (box_type, seed, zs, ngal, sl, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, sl, pickle.dumps(data)))
        self.conn.commit()

    def insert_noisy_patch(self, box_type, seed, zs, ngal, oa, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO noisy_patches (box_type, seed, zs, ngal, oa, data)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, oa, pickle.dumps(data)))
        self.conn.commit()

    def insert_smoothed_patch(self, box_type, seed, zs, ngal, sl, oa, data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO smoothed_patches (box_type, seed, zs, ngal, sl, oa, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, sl, oa, pickle.dumps(data)))
        self.conn.commit()

    def insert_noisy_map_stats(self, box_type, seed, zs, ngal, power_spectrum, pdf, peak, minima):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO noisy_map_stats (box_type, seed, zs, ngal, power_spectrum, pdf, peak, minima)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, pickle.dumps(power_spectrum), pickle.dumps(pdf), pickle.dumps(peak), pickle.dumps(minima)))
        self.conn.commit()

    def insert_smoothed_map_stats(self, box_type, seed, zs, ngal, sl, power_spectrum, pdf, peak, minima):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO smoothed_map_stats (box_type, seed, zs, ngal, sl, power_spectrum, pdf, peak, minima)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, sl, pickle.dumps(power_spectrum), pickle.dumps(pdf), pickle.dumps(peak), pickle.dumps(minima)))
        self.conn.commit()

    def insert_noisy_patch_stats(self, box_type, seed, zs, ngal, oa, bispectrum, power_spectrum, pdf, peak, minima):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO noisy_patch_stats (box_type, seed, zs, ngal, oa, bispectrum, power_spectrum, pdf, peak, minima)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, oa, pickle.dumps(bispectrum), pickle.dumps(power_spectrum), pickle.dumps(pdf), pickle.dumps(peak), pickle.dumps(minima)))
        self.conn.commit()

    def insert_smoothed_patch_stats(self, box_type, seed, zs, ngal, sl, oa, bispectrum, power_spectrum, pdf, peak, minima):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO smoothed_patch_stats (box_type, seed, zs, ngal, sl, oa, bispectrum, power_spectrum, pdf, peak, minima)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (box_type, seed, zs, ngal, sl, oa, pickle.dumps(bispectrum), pickle.dumps(power_spectrum), pickle.dumps(pdf), pickle.dumps(peak), pickle.dumps(minima)))
        self.conn.commit()

    def close(self):
        self.conn.close()
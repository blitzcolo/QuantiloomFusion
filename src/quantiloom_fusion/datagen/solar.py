"""Forcing-file reads and solar spectrum derivation: Planck-tail extension, forcing-matched and low-light scaling."""

import csv
import io
import math


def reference_direct(source, column):
    """Trapezoid integral (W/m2) of one irradiance column of a solar LUT.

    column is the renderer's 1-based solar_lut_columns index (1 = wavelength,
    so ASTM G-173's direct+circumsolar is 4). Integrated over the published
    table only: the Planck tail is a modelling extension, not the reference."""
    if type(column) is not int or column<2:raise ValueError(f'solar_lut column must be an integer >= 2, got {column!r}')
    with source.open(encoding="utf-8-sig") as f:
        rows=[row for row in csv.reader(f)][1:]
    values=[(float(row[0]),float(row[column-1])) for row in rows if row]
    if len(values)<2:raise ValueError(f'{source}: solar LUT needs at least two rows')
    return sum((w1-w0)*(e0+e1)/2 for (w0,e0),(w1,e1) in zip(values,values[1:]))


def forcing_direct(path, time_h):
    """Direct-beam irradiance (W/m2) of a thermal forcing CSV at time_h.

    Column 3 is sun_irradiance_w_m2 (time_h, air_temperature_k,
    sun_irradiance_w_m2, ...); comment lines start with '#', and a header is
    tolerated on the first line, as the renderer's loader does. Between rows
    the value is interpolated linearly."""
    rows=[]
    for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        line=line.strip()
        if not line or line.startswith('#'):continue
        cells=line.split(',')
        try:rows.append((float(cells[0]),float(cells[2])))
        except(ValueError,IndexError):
            if rows:raise ValueError(f'{path}: unreadable forcing row {i+1}: {line!r}')
    rows.sort()
    for (t0,e0),(t1,e1) in zip(rows,rows[1:]):
        if t0<=time_h<=t1:
            return e0 if t1==t0 else e0+(e1-e0)*(time_h-t0)/(t1-t0)
    if rows and rows[-1][0]==time_h:return rows[-1][1]
    raise ValueError(f'{path}: time_h {time_h} outside the forcing rows')


# Column order of a thermal forcing CSV as the renderer's loader reads it;
# the last two columns are optional.
FORCING_COLUMNS=('time_h','air_temperature_k','sun_irradiance_w_m2','sun_azimuth_deg','sun_elevation_deg',
                 'sky_temperature_k','diffuse_irradiance_w_m2','relative_humidity')


def forcing_row(path, time_h):
    """Every column of a thermal forcing CSV at time_h, interpolated linearly.

    Returns {column: value} for the columns both bracketing rows have, plus
    'source_lines': the 1-based line numbers of those rows (one number when
    time_h falls on a row)."""
    rows=[]
    for i,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        line=line.strip()
        if not line or line.startswith('#'):continue
        try:rows.append((float(line.split(',')[0]),[float(c) for c in line.split(',')],i+1))
        except ValueError:
            if rows:raise ValueError(f'{path}: unreadable forcing row {i+1}: {line!r}')
    rows.sort(key=lambda r:r[0])
    for (t0,a,l0),(t1,b,l1) in zip(rows,rows[1:]):
        if t0<=time_h<=t1:
            if time_h==t0 or t1==t0:values,lines=a,[l0]
            elif time_h==t1:values,lines=b,[l1]
            else:
                f=(time_h-t0)/(t1-t0);values=[x+(y-x)*f for x,y in zip(a,b)];lines=[l0,l1]
            return {**dict(zip(FORCING_COLUMNS,values)),'source_lines':lines}
    if len(rows)==1 and rows[0][0]==time_h:return {**dict(zip(FORCING_COLUMNS,rows[0][1])),'source_lines':[rows[0][2]]}
    raise ValueError(f'{path}: time_h {time_h} outside the forcing rows')


def solar_extension(source, destination, scale=1.0):
    # An explicitly labelled modelling assumption, not measured IR sunlight.
    with source.open(encoding="utf-8-sig") as f:
        rows=list(csv.reader(f))
    header=rows[0];values=[[float(v) for v in row] for row in rows[1:] if row]
    last=values[-1];anchor=last[0]
    def ratio(w):
        c2=1.438776877e7  # nm K
        return (anchor/w)**5*math.expm1(c2/(anchor*5778))/math.expm1(c2/(w*5778))
    if anchor<15000:
        for w in range(int(anchor)+100,15001,100):
            # Fixed formatting keeps the derived table byte-identical across OS libm.
            values.append([float(w),*[f'{v*ratio(w):.12e}' for v in last[1:]]])
    if scale!=1.0:
        # Low-light arm: every irradiance column is scaled together so the
        # direct/diffuse split the resolver derives is preserved. Tail rows
        # were already formatted, so the value goes back through float.
        values=[[row[0],*[f'{float(v)*scale:.12e}' for v in row[1:]]] for row in values]
    buffer=io.StringIO(newline='')
    writer=csv.writer(buffer);writer.writerow(header);writer.writerows(values)
    return buffer.getvalue()

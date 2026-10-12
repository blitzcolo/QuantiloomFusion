"""Solar spectrum derivation: Planck-tail extension and low-light scaling."""

import csv
import io
import math


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

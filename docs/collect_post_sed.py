import daf

MODULE = daf.__class__

def collect(prefix, module):
    for name in module.__all__ if hasattr(module, '__all__') else dir(module):
        if name[0] == "_":
            continue
        value = module.__dict__[name]
        if isinstance(value, MODULE):
            prefix.append(name)
            collect(prefix, value)
            prefix.pop()
            collect([f"_{name}"], value)
            continue
        prefix.append(name)
        full_name = ".".join(prefix)
        prefix.pop()
        print(f"s/>[']*{full_name}[']*/>{name}/g;")

collect(["daf"], daf)

print("s/anndata[.a-zA-Z0-9_]*AnnData/anndata.AnnData/g;")
print("s/numpy[.a-zA-Z0-9_]*ndarray/numpy.ndarray/g;")
print("s/numpy[.a-zA-Z0-9_]*dtype/numpy.dtype/g;")
print("s/pandas[.a-zA-Z0-9_]*DataFrame/pandas.DataFrame/g;")
print("s/pandas[.a-zA-Z0-9_]*Series/pandas.Series/g;")
print("s/scipy.sparse[.a-zA-Z0-9_]*coo_matrix/scipy.sparse.coo_matrix/g;")
print("s/scipy.sparse[.a-zA-Z0-9_]*csr_matrix/scipy.sparse.csc_matrix/g;")
print("s/scipy.sparse[.a-zA-Z0-9_]*csr_matrix/scipy.sparse.csr_matrix/g;")
print("s/scipy.sparse[.a-zA-Z0-9_]*spmatrix/scipy.sparse.spmatrix/g;")

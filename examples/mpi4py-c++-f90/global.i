%typemap(varout) double *dataPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    int dims[1];
    dims[0] = datacountPy*5;
    tmp = (PyArrayObject *)PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE
,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) double *dataPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}


%typemap(varout) int *iblPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    int dims[1];
    dims[0] = datacountPy;
    tmp = (PyArrayObject *)PyArray_FromDimsAndData(1,dims,PyArray_INT
,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) int *iblPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}



%typemap(varout) int *piloPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    int dims[1];
    dims[0] = 3;
    tmp = (PyArrayObject *)PyArray_FromDimsAndData(1,dims,PyArray_INT
,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) int *piloPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}

%typemap(varout) int *pihiPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    int dims[1];
    dims[0] = 3;
    tmp = (PyArrayObject *)PyArray_FromDimsAndData(1,dims,PyArray_INT
,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) int *pihiPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}

%typemap(varout) double *xloPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    int dims[1];
    dims[0] = 3;
    tmp = (PyArrayObject *)PyArray_FromDimsAndData(1,dims,PyArray_DOUBLE
,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) double *xloPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}










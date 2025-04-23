import operator
from typing import Iterable, List, Any, Union, Type, Dict, Tuple # Добавлены нужные типы
import numpy as np

# Список стандартных скалярных типов Python + NumPy (для is_scalar_type)
# Добавлен np.float64
scalar_types: List[type] = [int, float, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
                            np.float16, np.float32, np.float64, np.bool_]

# Список только скалярных типов NumPy (для get_np_scalar_types и др.)
# Добавлен np.float64
np_scalar_types: List[type] = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
                                np.float16, np.float32, np.float64, np.bool_]

# Словарь для маппинга NumPy dtypes в строки (похоже на OpenCL)
# Добавлен np.float64 -> 'double' (стандартное название в CL)
_np_dtype_to_cl: Dict[type, str] = {
    np.bool_: 'bool',
    np.uint8: 'uchar',
    np.int8: 'char',
    np.uint16: 'ushort',
    np.int16: 'short',
    np.uint32: 'uint',
    np.int32: 'int',
    np.uint64: 'ulong',
    np.int64: 'long',
    np.float16: 'half',
    np.float32: 'float',
    np.float64: 'double', # Добавлено
}

# Веса для определения "наиболее тяжелого" типа при кастинге/промоушене
# Добавлен np.float64
_np_dtype_weight: Dict[type, int] = {
    np.bool_: 1,
    np.uint8: 2,
    np.int8: 3,
    np.uint16: 4,
    np.int16: 5,
    np.uint32: 6,
    np.int32: 7,
    np.uint64: 8,
    np.int64: 9,
    np.float16: 10,
    np.float32: 11,
    np.float64: 12, # Добавлено
}

# Тип для представления элементов, которые может принять np.dtype()
DtypeLike = Union[str, type, np.dtype]
# Тип для представления элементов среза
SliceItem = Union[slice, int, type(Ellipsis), None]

class HType:
    """
    Helper functions for types.
    All methods are static as they don't depend on instance state.
    """

    @staticmethod
    def is_scalar_type(value: Any) -> bool:
        """Checks if the value's type is in the list of known scalar types (Python or NumPy)."""
        # Check instance's class directly
        return isinstance(value, tuple(scalar_types)) # More robust than value.__class__ in ...

    @staticmethod
    def get_np_scalar_types() -> List[type]:
        """Returns the list of supported NumPy scalar types."""
        return np_scalar_types

    @staticmethod
    def is_obj_of_np_scalar_type(obj: Any) -> bool:
        """Checks if the object's type is specifically one of the NumPy scalar types."""
        # Check instance's class directly
        return isinstance(obj, tuple(np_scalar_types)) # More robust than obj.__class__ in ...

    @staticmethod
    def np_dtype_to_cl(dtype: DtypeLike) -> str:
        """Converts a NumPy dtype (or dtype-like object) to its corresponding CL string."""
        try:
            # np.dtype(dtype).type gets the standard Python type object (e.g., np.float32)
            return _np_dtype_to_cl[np.dtype(dtype).type]
        except KeyError:
            raise TypeError(f"Unsupported numpy dtype for CL conversion: {dtype}")
        except Exception as e:
            raise TypeError(f"Could not process dtype '{dtype}': {e}")


    @staticmethod
    def get_most_weighted_dtype(dtype_list: Iterable[DtypeLike]) -> np.dtype:
        """Finds the dtype with the highest weight from a list of dtypes."""
        if not dtype_list:
            raise ValueError("Input dtype_list cannot be empty.")

        weighted_dtypes: List[Tuple[int, np.dtype]] = []
        for dt_like in dtype_list:
            try:
                dtype = np.dtype(dt_like)
                weight = _np_dtype_weight.get(dtype.type)
                if weight is None:
                    raise TypeError(f"Unsupported numpy dtype in list: {dt_like} (type: {dtype.type})")
                weighted_dtypes.append((weight, dtype))
            except Exception as e:
                 raise TypeError(f"Could not process dtype '{dt_like}' in list: {e}")


        # Sort by weight (descending)
        weighted_dtypes.sort(key=operator.itemgetter(0), reverse=True)
        # Return the dtype object (np.dtype) of the highest weighted item
        return weighted_dtypes[0][1]

    @staticmethod
    def hashable_slices(slices: Union[SliceItem, Iterable[SliceItem]]) -> tuple:
        """
        Convert a single slice item or an iterable of slice items into a hashable tuple.
        """
        # Ensure slices is iterable
        if not isinstance(slices, Iterable) or isinstance(slices, (str, bytes)): # str/bytes are iterable but usually not meant as list of slices
             _slices_iterable: Iterable[SliceItem] = (slices,)
        else:
            _slices_iterable = slices

        normalized_slices = []
        for x in _slices_iterable:
            if isinstance(x, slice):
                # Convert slice to a hashable tuple of its components
                normalized_slices.append( (slice, x.start, x.stop, x.step) ) # Add 'slice' marker for uniqueness
            elif isinstance(x, int):
                 # Keep integers as they are (hashable)
                normalized_slices.append(x)
            elif x is Ellipsis:
                # Keep Ellipsis (hashable)
                normalized_slices.append(Ellipsis) # Use the object itself
            elif x is None:
                 # Keep None (hashable)
                normalized_slices.append(None)
            else:
                # Handle potential unexpected types
                # Option 1: Raise error
                raise TypeError(f"Unhashable or unsupported type in slices: {type(x)}")
                # Option 2: Try to convert (original logic, less safe)
                # try:
                #     normalized_slices.append(int(x))
                # except (ValueError, TypeError):
                #     raise TypeError(f"Cannot convert item to int or handle type in slices: {type(x)}")

        return tuple(normalized_slices)


# Expose only HType at the module level
__all__ = ['HType']
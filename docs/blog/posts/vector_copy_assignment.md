---
date: 2025-07-19
categories:
  - Cpp
  - Vector
---

# 源码分析
<!-- more -->
```Cpp
_CONSTEXPR20 vector& operator=(const vector& _Right) {
    //自赋值检查：如果赋值的目标和源是同一个对象，就直接返回，不做任何操作
    if (this == _STD addressof(_Right)) {
        return *this;
    }

    //_Getal() 返回当前 vector 的 allocator（分配器）。
    //_Al 是当前对象的分配器，_Right_al 是源对象的分配器。
    auto& _Al       = _Getal();
    auto& _Right_al = _Right._Getal();
    if constexpr (_Choose_pocca_v<_Alty>) {//_Choose_pocca_v<_Alty> 是一个编译期布尔值，表示是否需要在 allocator 不同的时候清空对象
        if (_Al != _Right_al) {//如果分配器不相同
            _Tidy();//会销毁当前 vector 中的元素，并释放内存
            _Mypair._Myval2._Reload_proxy(_GET_PROXY_ALLOCATOR(_Alty, _Al), _GET_PROXY_ALLOCATOR(_Alty, _Right_al));
        }
    }

    _Pocca(_Al, _Right_al);//会在需要的时候把 _Right 的分配器拷贝到当前 vector。
    auto& _Right_data = _Right._Mypair._Myval2;
    _Assign_counted_range(_Right_data._Myfirst, static_cast<size_type>(_Right_data._Mylast - _Right_data._Myfirst));

    return *this;
}
```

```Cpp
//它负责把一段已知长度的区间（_First 到 _First + _Newsize）赋值给当前 vector。
template <class _Iter>
_CONSTEXPR20 void _Assign_counted_range(_Iter _First, const size_type _Newsize) {
    // assign elements from counted range _First + [0, _Newsize)
    auto& _Al         = _Getal();
    auto& _My_data    = _Mypair._Myval2;
    pointer& _Myfirst = _My_data._Myfirst;
    pointer& _Mylast  = _My_data._Mylast;
    pointer& _Myend   = _My_data._Myend;

    constexpr bool _Nothrow_construct = conjunction_v<is_nothrow_constructible<_Ty, _Iter_ref_t<_Iter>>,
        _Uses_default_construct<_Alloc, _Ty*, _Iter_ref_t<_Iter>>>;

    //使所有当前 vector 的迭代器失效（因为数据即将被替换）
    _My_data._Orphan_all();
    const auto _Oldcapacity = static_cast<size_type>(_Myend - _Myfirst);
    
    //如果新元素数量 > 旧容量：
    if (_Newsize > _Oldcapacity) {
        _Clear_and_reserve_geometric(_Newsize);//释放原来的所有元素,按几何增长规则（通常是 1.5x 或 2x）分配新的 buffer
        if constexpr (_Nothrow_construct) {
            _Mylast = _STD _Uninitialized_copy_n(_STD move(_First), _Newsize, _Myfirst, _Al);
            _ASAN_VECTOR_CREATE;
        } else {
            _ASAN_VECTOR_CREATE_GUARD;
            //在未构造内存中构造元素（不会先销毁旧元素，因为旧元素已在扩容时释放）。
            _Mylast = _STD _Uninitialized_copy_n(_STD move(_First), _Newsize, _Myfirst, _Al);
        }
        return;
    }

    const auto _Oldsize = static_cast<size_type>(_Mylast - _Myfirst);
    //新 size > 旧 size
    if (_Newsize > _Oldsize) {
        bool _Copied = false;
        if constexpr (_Iter_copy_cat<_Iter, pointer>::_Bitcopy_assignable) {
#if _HAS_CXX20
            if (!_STD is_constant_evaluated())
#endif // _HAS_CXX20
            {   
                //覆盖旧元素
                _Copy_memmove_n(_First, static_cast<size_t>(_Oldsize), _Myfirst);
                _First += _Oldsize;
                _Copied = true;
            }
        }

        if (!_Copied) {
            for (auto _Mid = _Myfirst; _Mid != _Mylast; ++_Mid, (void) ++_First) {
                *_Mid = *_First;
            }
        }

        if constexpr (_Nothrow_construct) {
            _ASAN_VECTOR_MODIFY(static_cast<difference_type>(_Newsize - _Oldsize));
            _Mylast = _STD _Uninitialized_copy_n(_STD move(_First), _Newsize - _Oldsize, _Mylast, _Al);
        } else {
            _ASAN_VECTOR_EXTEND_GUARD(_Newsize);
            //构造新元素
            _Mylast = _STD _Uninitialized_copy_n(_STD move(_First), _Newsize - _Oldsize, _Mylast, _Al);
            _ASAN_VECTOR_RELEASE_GUARD;
        }
    } else {
        const pointer _Newlast = _Myfirst + _Newsize;
        // 覆盖前 newsize 个元素
        _STD _Copy_n_unchecked4(_STD move(_First), _Newsize, _Myfirst);
        _STD _Destroy_range(_Newlast, _Mylast, _Al);
        // 销毁多余的尾部元素
        _ASAN_VECTOR_MODIFY(static_cast<difference_type>(_Newsize - _Oldsize));
        _Mylast = _Newlast;
    }
}
```
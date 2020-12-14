/*
  Template stuff for type-promotion

  needed by threevector and simplefuncs

  Time-stamp: <promote_numeric.h on Saturday, 8 September, 2012 at 16:02:45 MST (philip)>
 */

#ifndef __PROMOTE_NUMERIC_H_INCLUDED__
#define __PROMOTE_NUMERIC_H_INCLUDED__
#include <limits>

template <bool, class T, class U>
struct SelectIf {};

template <class T, class U>
struct SelectIf<true, T, U> { typedef T type; };

template <class T, class U>
struct SelectIf<false, T, U> { typedef U type; };

template <class T, class U>
struct PromoteNumeric {
    typedef typename SelectIf<
        //if T and U are both integers or both non-integers
        std::numeric_limits<T>::is_integer == std::numeric_limits<U>::is_integer,
        //then pick the larger type
        typename SelectIf<(sizeof(T) > sizeof(U)), T,
        //else if they are equal
        typename SelectIf<(sizeof(T) == sizeof(U)),
                          //pick the one which is unsigned
                          typename SelectIf<std::numeric_limits<T>::is_signed, U, T>::type,
                          //else select U as bigger
                          U
                          >::type
        >::type,
    //else pick the one which is not integer
        typename SelectIf<std::numeric_limits<T>::is_integer, U, T>::type
        >::type type;
};

#endif // __PROMOTE_NUMERIC_H_INCLUDED__

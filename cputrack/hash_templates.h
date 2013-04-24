#pragma once
#include <hash_map>
#include <hash_set>

namespace qtrk {
#ifdef _MSC_VER
	template<typename TKey, typename T>
	class hash_map : public stdext::hash_map<TKey, T> {};
	template<typename T>
	class hash_set : public stdext::hash_set<T> {};
#else
	template<typename TKey, typename T>
	class hash_map : public __gnu_cxx::hash_map<TKey, T> {};
	template<typename T>
	class hash_set : public __gnu_cxx::hash_set<T> {};
#endif
};

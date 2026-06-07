if GROUP == "All" || GROUP == "Downstream"
    @safetestset "Modeling Toolkit Cache Indexing" include("mtk_cache_indexing_tests__item1.jl")
end

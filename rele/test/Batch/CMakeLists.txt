#Batch test

add_executable(deep_off DeepBatchTest.cpp)
target_link_libraries(deep_off rele)

add_executable(nls_off NlsBatchTest.cpp)
target_link_libraries(nls_off rele)

add_executable(portfolio_off PortfolioBatchTest.cpp)
target_link_libraries(portfolio_off rele)

add_executable(pend PendulumTest.cpp)
target_link_libraries(pend rele)

add_executable(mc_GPFQI McGPFQITest.cpp)
target_link_libraries(mc_GPFQI rele)

add_executable(mc_LSPI McLSPITest.cpp)
target_link_libraries(mc_LSPI rele)

add_executable(GPFQI_policy GP-FQIPolicyTest.cpp)
target_link_libraries(GPFQI_policy rele)

add_executable(ContinuousSwingUpTest ContinuousSwingUpTest.cpp)
target_link_libraries(ContinuousSwingUpTest rele)

add_executable(GP-ContinuousFQIPolicyTest GP-ContinuousFQIPolicyTest.cpp)
target_link_libraries(GP-ContinuousFQIPolicyTest rele)


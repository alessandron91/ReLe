#TD tests
add_executable(simple_chain SimpleChain.cpp)
target_link_libraries(simple_chain rele)

add_executable(simple_chain_mean SimpleChainMeanReward.cpp)
target_link_libraries(simple_chain_mean rele)

add_executable(grid_world GridWorldTest.cpp)
target_link_libraries(grid_world rele)

add_executable(grid_world2 GridWorldTest2.cpp)
target_link_libraries(grid_world2 rele)

add_executable(mountain_car MountainCarTest.cpp)
target_link_libraries(mountain_car rele)

add_executable(forex ForexTest.cpp)
target_link_libraries(forex rele)

add_executable(swing_pendulum SwingPendulumTest.cpp)
target_link_libraries(swing_pendulum rele)

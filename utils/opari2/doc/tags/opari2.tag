<?xml version='1.0' encoding='ISO-8859-1' standalone='yes' ?>
<tagfile>
  <compound kind="page">
    <name>index</name>
    <title>Opari2</title>
    <filename>index</filename>
    <docanchor file="index">pomp_tpd</docanchor>
    <docanchor file="index">NEWS</docanchor>
    <docanchor file="index">LINKING</docanchor>
    <docanchor file="index">POMP_user_instrumentation</docanchor>
    <docanchor file="index">TASKING</docanchor>
    <docanchor file="index">EXAMPLE</docanchor>
    <docanchor file="index">INSTALLATION</docanchor>
    <docanchor file="index">SUMMARY</docanchor>
    <docanchor file="index">POMP2_Parallel_fork</docanchor>
    <docanchor file="index">POMP2</docanchor>
    <docanchor file="index">LINK_STEP</docanchor>
    <docanchor file="index">CTC_STRING</docanchor>
    <docanchor file="index">USAGE</docanchor>
  </compound>
  <compound kind="file">
    <name>pomp2_lib.h</name>
    <path>/home/roessel/silc/opari2/src/tags/REL-1.0.2/include/opari2/</path>
    <filename>pomp2__lib_8h</filename>
    <member kind="typedef">
      <type>void *</type>
      <name>POMP2_Region_handle</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a6eab8920b6cebf312eba77f9892d133e</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Assign_handle</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a4ced2800b4c94cb22193e4a8b9d7a0b2</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Atomic_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a531dca8864ae213bbc7d94c5cb961636</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Atomic_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a3afbaf1c26bba2d3684fed7267da2b43</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Barrier_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a7d6c46f9cafa0dd8e27ac714ae5f9c1c</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle *pomp2_old_task, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Barrier_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a545aa07b70b437e18b72804f944ebaa3</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af8064076acfc2edf88aa67343cdbe063</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Critical_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a54ed08ae5c5e0a08873e389e4b52065f</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Critical_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af4964bfdd1ed19bdbd892e294505740a</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Critical_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>aa93bad98f0b6ae5ed98fd55247590dfa</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Critical_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>aa5721b96b51f2d283f6ccf266e13c1ef</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Destroy_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a8f80c9df188425c5ee16a555acd65b15</anchor>
      <arglist>(omp_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Destroy_nest_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a19f5e66b420c766bea8e72bd1cd82849</anchor>
      <arglist>(omp_nest_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_End</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ad05ed5d28dc0294b97b99b5db935fb62</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Finalize</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a4577c9e151c40c2319a8b423bd55f487</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Flush_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a6fab310b87ca2e43b101d1835835f26e</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Flush_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>aa495f0ad5561de54a2010597dfff6ffe</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_For_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a74e9a19ea05e07fe8223eabaae796b8d</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_For_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a0ff4d4a1227f23c9124088be417b2822</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>POMP2_Task_handle</type>
      <name>POMP2_Get_new_task_handle</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a0f86501cade4447ea61f8672cbd2f13f</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Implicit_barrier_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>adcd2a8541aec9be09edf41cdf1afccba</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle *pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Implicit_barrier_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a622ac0d092b107eb41d40d916a5634c7</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Init</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a934d6182d758d9bba0fa3fdcdb89cbec</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Init_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a1b5645df7e69d60b1fce4ab29493a2d9</anchor>
      <arglist>(omp_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Init_nest_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a4c3778052086779ea27b445fe43b0670</anchor>
      <arglist>(omp_nest_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>POMP2_Lib_get_max_threads</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a22e83c1e99d987da0fccf1ed1edd9836</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Master_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ae7a04a00e9c09494befe0ea8ff254cdc</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Master_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a2fbdfc4a57643910d0d773838e1c1099</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Off</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ae0a417aab012ce3b5f4354afd334b549</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_On</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a961e240ee2973edcf4d64fe02bf64ac6</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Ordered_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a87fbea1fe64064d3ed61042e34f62c12</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Ordered_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>aa87692ea968d546c695bb1a320e1ec67</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Ordered_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a2353ea2fd741887a13fc926c32e39abd</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Ordered_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a01bda2ec53cce003b9dc71b202f235a7</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Parallel_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a81eb542b411bd83ab0e9999c1db43a53</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Parallel_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>abfff83102191df12c6cc8e31f05d230b</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Parallel_fork</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ab573944b1528335208d4341192a05312</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, int if_clause, int num_threads, POMP2_Task_handle *pomp2_old_task, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Parallel_join</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a77bceb911f3eef2d1ad2333f22de91ca</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Section_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ab887e462c792fa60048dafe8d71de3b3</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Section_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ade10ed9951c5635d995be1f1983c13e5</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Sections_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ad4f841e493403a75b10f66b963fb67fc</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Sections_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a70ca605e4bad5b164e5328fcd317ab40</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Set_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ab667c8dbd0f4fe8946690e9e74dbcf7d</anchor>
      <arglist>(omp_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Set_nest_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af87fb20e0630b6356bd566d100ea85c9</anchor>
      <arglist>(omp_nest_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Single_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ac5e6f8ff1b48c9486570fa4852696bcf</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Single_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ae796aa146d8c483b23ca1293bb9df1aa</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Single_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af3dc721051567aed79705643accd92d3</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Single_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ac0316c98966e681739214bd914c8380a</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Task_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a632472236899de4fd592877cabd81517</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Task_create_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a1604404e7509bfd0936019101ad7d341</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle *pomp2_new_task, POMP2_Task_handle *pomp2_old_task, int pomp2_if, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Task_create_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a8c7c300057b2c8b4093a70359359dc9a</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Task_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a13ce037fa0139bd0746feb1e41f1f309</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Taskwait_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a3831d9e71f12714054be84dc69ad0f4c</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle *pomp2_old_task, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Taskwait_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a6c09877b319ec68e49859be35a3b7547</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>POMP2_Test_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a59a92f2604851843783a25ac738d6048</anchor>
      <arglist>(omp_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>int</type>
      <name>POMP2_Test_nest_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a01e999347398c0812565af0ef871a1db</anchor>
      <arglist>(omp_nest_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Unset_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ab90737dacdfdbf2e14390398f8610b00</anchor>
      <arglist>(omp_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Unset_nest_lock</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>abf1399e2dbd818523670dbd369d7de9e</anchor>
      <arglist>(omp_nest_lock_t *s)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Untied_task_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a740487f88b52e63d4526fc1e07d59ae0</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Untied_task_create_begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a6f21ea34669ca63d3a138a2bac973c78</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle *pomp2_new_task, POMP2_Task_handle *pomp2_old_task, int pomp2_if, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Untied_task_create_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a5a5199b161b3fe470990f554b9c8435e</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, POMP2_Task_handle pomp2_old_task)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Untied_task_end</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af423fd9fdd8eb4e755548518e56638f6</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Workshare_enter</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a57cc5fabf079df983ea2522ee40e0eb5</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle, const char ctc_string[])</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Workshare_exit</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a24cd4bda2b8a17ee7d675467d138fd68</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>POMP2_Get_num_regions</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a19ef196d4c3eead3fa5e50bf74318db3</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>POMP2_Init_regions</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a7b27c3ff4d1141991864e7ace992f057</anchor>
      <arglist>()</arglist>
    </member>
    <member kind="function">
      <type>const char *</type>
      <name>POMP2_Get_opari2_version</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>ac1faad005fc7715e93bf12a802866eda</anchor>
      <arglist>()</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>pomp2_region_info.h</name>
    <path>/home/roessel/silc/opari2/src/tags/REL-1.0.2/src/pomp-lib-dummy/</path>
    <filename>pomp2__region__info_8h</filename>
    <class kind="struct">POMP2_Region_info</class>
    <member kind="enumeration">
      <name>POMP2_Region_type</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>a680a7412e9daca09b793c6538fb7b381</anchor>
      <arglist></arglist>
    </member>
    <member kind="enumeration">
      <name>POMP2_Schedule_type</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>a9bada01c672e9a100dc7903ba06e76a7</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>ctcString2RegionInfo</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>ada8fce980385bdc6598a889bc2ba7892</anchor>
      <arglist>(const char ctcString[], POMP2_Region_info *regionInfo)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>freePOMP2RegionInfoMembers</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>a328c8dd30c3edae6a9c076f990ea36c6</anchor>
      <arglist>(POMP2_Region_info *regionInfo)</arglist>
    </member>
    <member kind="function">
      <type>const char *</type>
      <name>pomp2RegionType2String</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>a41324a61d69375fcf4d905a2db32a1b5</anchor>
      <arglist>(POMP2_Region_type regionType)</arglist>
    </member>
    <member kind="function">
      <type>const char *</type>
      <name>pomp2ScheduleType2String</name>
      <anchorfile>pomp2__region__info_8h.html</anchorfile>
      <anchor>a5a45288617878806cd3773be2bad8ec4</anchor>
      <arglist>(POMP2_Schedule_type scheduleType)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>POMP2_Region_info</name>
    <filename>structPOMP2__Region__info.html</filename>
    <member kind="variable">
      <type>POMP2_Region_type</type>
      <name>mRegionType</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a34a3ca3ada9bbf45ae4f1e8248b13e70</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>char *</type>
      <name>mStartFileName</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a6034a8b8c7b55bc7979977b9334958db</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned</type>
      <name>mStartLine1</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a9f0d2a2d58e6ee5e23b2b1e0a1d4ebc1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned</type>
      <name>mStartLine2</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a5685e6673c14de2660bf224eaa554b76</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>char *</type>
      <name>mEndFileName</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a8c0105ca03822a142662e80953d75358</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned</type>
      <name>mEndLine1</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a94d58f08bbf8e8529fedb7cbccffd017</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned</type>
      <name>mEndLine2</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>ade27fd75d1537732a2459c78f818eb8f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasCopyIn</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a2f7f3d1dc98587dbff386c6d8e38b96b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasCopyPrivate</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a4bf7f82ec65b0115b0a929e307c1bf58</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasIf</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>afaf850aae51e3e80f1566ee8fbf836f4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasFirstPrivate</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a4087657089faf52b1d6edc383388ff85</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasLastPrivate</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a8e6ec810e13638ac587810f23678487c</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasNoWait</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a5f2a12e99defcacafcba38d7c161e89d</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasNumThreads</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>af9ecc5af683aba508fb892b2738bbdc5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasOrdered</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a24a152ec56e0eafe41326d1d0329bc9b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasReduction</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a629015eae4cec0d4bc7e0da0f0dfaeff</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasCollapse</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a80c536c6186f3c61f0358b073a302571</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>mHasUntied</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a8790f498cf86eb6808a9df5d01449677</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>POMP2_Schedule_type</type>
      <name>mScheduleType</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>aeaa2f47c4731b4e10dd3bfb35a2024a3</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>char *</type>
      <name>mUserGroupName</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>ae8cebc481d4414f0857d96080ca16a8a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>unsigned</type>
      <name>mNumSections</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a63d659e3d853a16e7dc2364a345af231</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>char *</type>
      <name>mCriticalName</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>a4e4565e6d2c2881d007ab343769f6766</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>char *</type>
      <name>mUserRegionName</name>
      <anchorfile>structPOMP2__Region__info.html</anchorfile>
      <anchor>abb1c3408c4239654c970ec2e08d49d4c</anchor>
      <arglist></arglist>
    </member>
  </compound>
</tagfile>

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
    <path>/home/roessel/silc/opari2/src/tags/REL-1.0.7-rc1/include/opari2/</path>
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
      <name>POMP2_Begin</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>af8064076acfc2edf88aa67343cdbe063</anchor>
      <arglist>(POMP2_Region_handle *pomp2_handle)</arglist>
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
      <type>POMP2_Task_handle</type>
      <name>POMP2_Get_new_task_handle</name>
      <anchorfile>pomp2__lib_8h.html</anchorfile>
      <anchor>a0f86501cade4447ea61f8672cbd2f13f</anchor>
      <arglist>()</arglist>
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
    <path>/home/roessel/silc/opari2/src/tags/REL-1.0.7-rc1/src/pomp-lib-dummy/</path>
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
  <compound kind="page">
    <name>installationfile</name>
    <title>OPARI2 INSTALL</title>
    <filename>installationfile</filename>
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

/**
 * DrugEx Hub — Master Multi-Module Lecture Aggregate Database (18 Comprehensive Lectures)
 * Tailored for Bachelor's Thesis on De Novo Drug Design & ROCS Shape Matching for Flexible Targets / IDPs
 */

import { M1_LECTURES } from "./lectures/m1.js";
import { M2_LECTURES } from "./lectures/m2.js";
import { M3_LECTURES } from "./lectures/m3.js";
import { M4_LECTURES } from "./lectures/m4.js";
import { M5_LECTURES } from "./lectures/m5.js";
import { M6_LECTURES } from "./lectures/m6.js";

export const LECTURE_DATA = {
  ...M1_LECTURES,
  ...M2_LECTURES,
  ...M3_LECTURES,
  ...M4_LECTURES,
  ...M5_LECTURES,
  ...M6_LECTURES
};

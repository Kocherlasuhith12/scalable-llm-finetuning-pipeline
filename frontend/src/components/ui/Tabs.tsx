"use client";

import React from "react";
import { motion } from "framer-motion";

export interface TabItem {
  id: string;
  label: string;
  count?: number;
}

export interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onChange, className = "" }) => {
  return (
    <div className={`flex items-center gap-1 p-1 bg-[#121216] border border-[#2A2A35] rounded-xl w-fit ${className}`}>
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className="relative px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all select-none flex items-center gap-2 cursor-pointer"
          >
            {isActive && (
              <motion.div
                layoutId="activeTabPill"
                className="absolute inset-0 bg-[#E11D48]/20 border border-[#E11D48]/40 rounded-lg shadow-[0_0_12px_rgba(225,29,72,0.15)]"
                transition={{ type: "spring", stiffness: 500, damping: 35 }}
              />
            )}
            <span className={`relative z-10 ${isActive ? "text-[#F43F5E]" : "text-[#94A3B8] hover:text-[#F8FAFC]"}`}>
              {tab.label}
            </span>
            {tab.count !== undefined && (
              <span
                className={`relative z-10 px-1.5 py-0.2 text-[10px] rounded-full ${
                  isActive ? "bg-[#E11D48] text-white" : "bg-[#2A2A35] text-[#94A3B8]"
                }`}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};

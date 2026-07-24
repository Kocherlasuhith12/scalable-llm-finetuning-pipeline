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
    <div className={`flex items-center gap-1 p-1 bg-[#111318] border border-[#27272A] rounded-xl w-fit ${className}`}>
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
                className="absolute inset-0 bg-[#8B5CF6]/20 border border-[#8B5CF6]/40 rounded-lg"
                transition={{ type: "spring", stiffness: 500, damping: 35 }}
              />
            )}
            <span className={`relative z-10 ${isActive ? "text-[#A78BFA]" : "text-[#94A3B8] hover:text-[#F8FAFC]"}`}>
              {tab.label}
            </span>
            {tab.count !== undefined && (
              <span
                className={`relative z-10 px-1.5 py-0.2 text-[10px] rounded-full ${
                  isActive ? "bg-[#8B5CF6] text-white" : "bg-[#27272A] text-[#94A3B8]"
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

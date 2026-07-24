"use client";

import React from "react";
import { ChevronRight, Search, Bell, Plus, Sparkles } from "lucide-react";
import { Button } from "./Button";

interface TopbarProps {
  activeTab: string;
  onOpenCmdK: () => void;
  onNewWorkload?: () => void;
}

export const Topbar: React.FC<TopbarProps> = ({ activeTab, onOpenCmdK, onNewWorkload }) => {
  return (
    <header className="h-16 border-b border-[#27272A] bg-[#111318]/80 backdrop-blur-xl sticky top-0 z-30 px-8 flex items-center justify-between select-none">
      {/* Breadcrumb Navigation */}
      <div className="flex items-center gap-2 text-xs text-[#94A3B8]">
        <span className="font-medium hover:text-[#F8FAFC] transition-colors cursor-pointer">HyperTune AI</span>
        <ChevronRight className="w-3.5 h-3.5 text-[#64748B]" />
        <span className="font-semibold text-[#F8FAFC] capitalize tracking-wide">{activeTab}</span>
      </div>

      {/* Action Controls & Global Search */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenCmdK}
          className="flex items-center gap-3 px-3.5 py-1.5 bg-[#171A21]/90 border border-[#27272A] rounded-lg text-xs text-[#94A3B8] hover:border-[#8B5CF6]/50 hover:text-[#F8FAFC] transition-all w-60 cursor-pointer shadow-sm"
        >
          <Search className="w-3.5 h-3.5 text-[#64748B]" />
          <span>Search platform...</span>
          <span className="ml-auto text-[10px] font-mono px-1.5 py-0.5 bg-[#09090B] border border-[#27272A] rounded text-[#64748B]">
            ⌘K
          </span>
        </button>

        <button className="w-8 h-8 rounded-lg bg-[#171A21]/90 border border-[#27272A] flex items-center justify-center text-[#94A3B8] hover:text-[#F8FAFC] hover:border-[#3f3f46] transition-all relative cursor-pointer shadow-sm">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-[#8B5CF6] rounded-full shadow-[0_0_8px_#8B5CF6]"></span>
        </button>

        {onNewWorkload && (
          <Button
            variant="primary"
            size="sm"
            onClick={onNewWorkload}
            leftIcon={<Plus className="w-3.5 h-3.5" />}
          >
            New Workload
          </Button>
        )}
      </div>
    </header>
  );
};

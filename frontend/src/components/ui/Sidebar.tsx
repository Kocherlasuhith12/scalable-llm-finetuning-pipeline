"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LucideIcon,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Search,
  Check,
  Building2,
  Server,
  FlaskConical,
  LogOut,
  Sparkles
} from "lucide-react";
import { Tooltip } from "./Tooltip";
import { Dropdown, DropdownItem, DropdownHeader, DropdownDivider } from "./Dropdown";

export interface NavItem {
  id: string;
  label: string;
  icon: LucideIcon;
  count?: number;
}

interface SidebarProps {
  navItems: NavItem[];
  activeTab: string;
  onTabChange: (id: string) => void;
  userName?: string;
  userEmail?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  navItems,
  activeTab,
  onTabChange,
  userName = "Admin Account",
  userEmail = "admin@enterprise.ai"
}) => {
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [activeWorkspace, setActiveWorkspace] = useState({
    id: "prod",
    name: "Acme Enterprise",
    subtitle: "Production Workspace",
    icon: Building2
  });

  const workspaces = [
    { id: "prod", name: "Acme Enterprise", subtitle: "Production Workspace", icon: Building2 },
    { id: "staging", name: "Staging Cluster", subtitle: "US-East (N. Virginia)", icon: Server },
    { id: "research", name: "Research Lab", subtitle: "Experimental H100 Cluster", icon: FlaskConical }
  ];

  const filteredNavItems = navItems.filter((item) =>
    item.label.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <motion.aside
      initial={false}
      animate={{ width: isCollapsed ? 80 : 256 }}
      transition={{ type: "spring", stiffness: 350, damping: 30 }}
      className="bg-[#111318] border-r border-[#27272A] flex flex-col p-3 z-40 select-none relative h-screen sticky top-0"
    >
      {/* Workspace Switcher Header */}
      <div className="mb-4">
        <Dropdown
          align="left"
          className="w-60"
          trigger={
            <div
              className={`flex items-center justify-between p-2.5 bg-[#171A21] border border-[#27272A] rounded-xl hover:border-[#8B5CF6]/50 transition-all cursor-pointer ${
                isCollapsed ? "justify-center" : ""
              }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#8B5CF6] to-[#6D28D9] flex items-center justify-center font-bold text-xs text-white shadow-md shadow-[#8B5CF6]/20 flex-shrink-0">
                  <activeWorkspace.icon className="w-4 h-4 text-white" />
                </div>
                {!isCollapsed && (
                  <div className="min-w-0 flex-1">
                    <div className="font-semibold text-xs text-[#F8FAFC] truncate">
                      {activeWorkspace.name}
                    </div>
                    <div className="text-[10px] text-[#94A3B8] truncate">
                      {activeWorkspace.subtitle}
                    </div>
                  </div>
                )}
              </div>
              {!isCollapsed && <ChevronDown className="w-3.5 h-3.5 text-[#64748B] flex-shrink-0" />}
            </div>
          }
        >
          <DropdownHeader>Switch Workspace</DropdownHeader>
          {workspaces.map((ws) => {
            const IconComp = ws.icon;
            const isSelected = activeWorkspace.id === ws.id;
            return (
              <DropdownItem
                key={ws.id}
                onClick={() => setActiveWorkspace(ws)}
                icon={<IconComp className="w-4 h-4" />}
                shortcut={isSelected ? "Active" : undefined}
              >
                <span>{ws.name}</span>
              </DropdownItem>
            );
          })}
          <DropdownDivider />
          <DropdownItem danger icon={<LogOut className="w-4 h-4" />}>
            Sign Out
          </DropdownItem>
        </Dropdown>
      </div>

      {/* In-sidebar Search (Visible when expanded) */}
      {!isCollapsed && (
        <div className="mb-4 px-1">
          <div className="relative flex items-center">
            <Search className="w-3.5 h-3.5 absolute left-3 text-[#64748B] pointer-events-none" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Quick filter..."
              className="w-full bg-[#09090B] border border-[#27272A] rounded-lg pl-8 pr-3 py-1.5 text-xs text-[#F8FAFC] placeholder-[#64748B] focus:outline-none focus:border-[#8B5CF6] transition-all"
            />
          </div>
        </div>
      )}

      {/* Navigation List */}
      <nav className="flex-1 space-y-1 overflow-y-auto pr-1">
        {!isCollapsed && (
          <div className="px-3 py-2 text-[10px] font-bold tracking-wider text-[#64748B] uppercase">
            Platform Modules
          </div>
        )}

        {filteredNavItems.map((item) => {
          const IconComponent = item.icon;
          const isActive = activeTab === item.id;

          const buttonContent = (
            <button
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center justify-between ${
                isCollapsed ? "px-0 justify-center h-10" : "px-3 py-2.5"
              } rounded-lg text-xs font-medium transition-all relative cursor-pointer ${
                isActive
                  ? "bg-[#8B5CF6]/15 text-[#A78BFA]"
                  : "text-[#94A3B8] hover:bg-[#171A21] hover:text-[#F8FAFC]"
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebarActivePill"
                  className="absolute left-0 top-1 bottom-1 w-1 bg-[#8B5CF6] rounded-r-full shadow-[0_0_10px_#8B5CF6]"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
              <div className="flex items-center gap-2.5">
                <IconComponent className={`w-4 h-4 ${isActive ? "text-[#8B5CF6]" : "text-[#64748B]"}`} />
                {!isCollapsed && <span>{item.label}</span>}
              </div>

              {!isCollapsed && item.count !== undefined && item.count > 0 && (
                <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-[#8B5CF6]/20 text-[#A78BFA]">
                  {item.count}
                </span>
              )}
            </button>
          );

          if (isCollapsed) {
            return (
              <Tooltip key={item.id} content={item.label} position="right">
                {buttonContent}
              </Tooltip>
            );
          }

          return <div key={item.id}>{buttonContent}</div>;
        })}
      </nav>

      {/* Footer Profile & Collapse Toggle */}
      <div className="pt-3 border-t border-[#27272A] space-y-3">
        {!isCollapsed && (
          <div className="flex items-center justify-between px-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-7 h-7 rounded-full bg-[#8B5CF6] flex items-center justify-center font-bold text-xs text-white shadow-sm flex-shrink-0">
                {userName.charAt(0)}
              </div>
              <div className="min-w-0">
                <div className="text-xs font-semibold text-[#F8FAFC] truncate">{userName}</div>
                <div className="text-[10px] text-[#94A3B8] truncate">{userEmail}</div>
              </div>
            </div>
          </div>
        )}

        <button
          onClick={() => setIsCollapsed((prev) => !prev)}
          className="w-full flex items-center justify-center p-2 rounded-lg bg-[#171A21] border border-[#27272A] text-[#94A3B8] hover:text-[#F8FAFC] hover:border-[#8B5CF6]/40 transition-all cursor-pointer text-xs gap-2"
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span>Collapse Sidebar</span>
            </>
          )}
        </button>
      </div>
    </motion.aside>
  );
};

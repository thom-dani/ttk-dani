// TTK Welcome message.
// Julien Tierny <julien.tierny@sorbonne-universite.fr>
// January 2020.

// "TTK                   (c) 2025"

printMsg(
  debug::output::BOLD
    + " _____ _____ _  __                    __  __    ____   ___ ____  ____"
    + debug::output::ENDCOLOR,
  debug::Priority::PERFORMANCE,
  debug::LineMode::NEW,
  stream);
printMsg(debug::output::BOLD
           + "|_   _|_   _| |/ /                   / /__\\ \\  |___ \\ / _ "
             "\\___ \\| ___|"
           + debug::output::ENDCOLOR,
         debug::Priority::PERFORMANCE,
         debug::LineMode::NEW,
         stream);
printMsg(
  debug::output::BOLD
    + "  | |   | | | ' /                   | |/ __| |   __) | | | |__) |___ \\"
    + debug::output::ENDCOLOR,
  debug::Priority::PERFORMANCE,
  debug::LineMode::NEW,
  stream);
printMsg(
  debug::output::BOLD
    + "  | |   | | | . \\                   | | (__| |  / __/| |_| / __/ ___) |"
    + debug::output::ENDCOLOR,
  debug::Priority::PERFORMANCE,
  debug::LineMode::NEW,
  stream);
printMsg(debug::output::BOLD
           + "  |_|   |_| |_|\\_\\                  | |\\___| | "
             "|_____|\\___/_____|____/"
           + debug::output::ENDCOLOR,
         debug::Priority::PERFORMANCE,
         debug::LineMode::NEW,
         stream);
printMsg(debug::output::BOLD + "                                     \\_\\  /_/"
           + debug::output::ENDCOLOR,
         debug::Priority::PERFORMANCE,
         debug::LineMode::NEW,
         stream);

printMsg(debug::output::BOLD + "Welcome!" + debug::output::ENDCOLOR,
         debug::Priority::PERFORMANCE,
         debug::LineMode::NEW,
         stream);

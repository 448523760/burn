# dataset

## Dataset

- Dataset<I>: Send + Sync

  Send 和 Sync: 这是 Rust 中的 trait，用于指示类型的并发安全性。
    - Send trait 表示该类型可以安全地在线程之间传递所有权。具有 Send trait 的类型是线程安全的，可以跨线程传递，因为它们保证了没有数据竞争的情况。
    - Sync trait 表示该类型可以安全地在多个线程之间共享可变的访问。具有 Sync trait 的类型是线程安全的，并且可以被多个线程同时访问，但只能是不可变访问。

Dataset trait 要求实现它的类型**既要是可以在不同线程之间传递所有权的（Send trait）**，**又要是可以在多个线程之间共享可变访问的（Sync trait）**。

